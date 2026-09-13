"""Read-only full-scene/model acquisition pumped by the existing ROS node owner.

No spin, service wait, model loading, actuation or admission lives here. The two
services are independent reads, not an atomic Scene/model transaction; the caller
still owns the resulting model/Scene/revision and current-execution checks.
"""
import copy
import math
import time

from tools.fr5_data_factory import ContractError


class GeometryCapture:
    """One capture, original deadline, two owned clients and at most two futures."""

    def __init__(self, node, *, deadline, clock=time.monotonic):
        from moveit_msgs.msg import PlanningSceneComponents
        from moveit_msgs.srv import GetPlanningScene
        from rcl_interfaces.srv import GetParameters

        self._node, self._clock = node, clock
        self._started = self._now()
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or deadline <= self._started):
            raise ContractError("GEOMETRY_CAPTURE_DEADLINE")
        self._deadline = deadline
        self._closed = False
        self._returned = False
        components = PlanningSceneComponents
        scene = GetPlanningScene.Request()
        scene.components.components = (
            components.SCENE_SETTINGS | components.ROBOT_STATE | components.ROBOT_STATE_ATTACHED_OBJECTS
            | components.WORLD_OBJECT_NAMES | components.WORLD_OBJECT_GEOMETRY | components.OCTOMAP
            | components.TRANSFORMS | components.ALLOWED_COLLISION_MATRIX | components.LINK_PADDING_AND_SCALING
            | components.OBJECT_COLORS)
        parameters = GetParameters.Request(names=["robot_description", "robot_description_semantic"])
        self._reads = [dict(kind=kind, endpoint=endpoint, request=request, client=None, future=None)
                       for kind, endpoint, request in (
                           (GetPlanningScene, "/get_planning_scene", scene),
                           (GetParameters, "/move_group/get_parameters", parameters))]

    def _now(self):
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ContractError("GEOMETRY_CAPTURE_CLOCK")
        return value

    def _check_deadline(self):
        now = self._now()
        if now < self._started:
            raise ContractError("GEOMETRY_CAPTURE_CLOCK")
        if now >= self._deadline:
            raise ContractError("GEOMETRY_CAPTURE_TIMEOUT")

    def poll(self):
        """Return one detached {scene, urdf, srdf}, or None without blocking.

        Client creation is deferred until ownership of this object is acquired,
        so partial setup failure remains accessible to close(). No caller spins
        this node concurrently: its existing callback pump progresses futures.
        """
        if self._closed or self._returned:
            return None
        try:
            self._check_deadline()
            for read in self._reads:
                if self._closed:
                    return None
                if read["client"] is None:
                    read["client"] = self._node.create_client(read["kind"], read["endpoint"])
                    if self._closed:
                        self.close()
                        return None
                self._check_deadline()
                if read["future"] is None:
                    available = read["client"].service_is_ready()
                    self._check_deadline()
                    if self._closed:
                        return None
                    if available:
                        read["future"] = read["client"].call_async(read["request"])
                        if self._closed:
                            self.close()
                            return None
            self._check_deadline()
            futures = [read["future"] for read in self._reads]
            if any(future is not None and future.cancelled() for future in futures):
                raise ContractError("GEOMETRY_CAPTURE_CANCELLED")
            if any(future is None or not future.done() for future in futures):
                return None
            values = []
            for future in futures:
                if self._closed:
                    return None
                values.append(future.result())  # done() checked; never a wait.
            from moveit_msgs.srv import GetPlanningScene
            from rcl_interfaces.msg import ParameterType
            from rcl_interfaces.srv import GetParameters

            scene, parameters = values
            if (not isinstance(scene, GetPlanningScene.Response) or scene.scene.is_diff is not False
                    or not scene.scene.robot_model_name.strip()
                    or not isinstance(parameters, GetParameters.Response) or len(parameters.values) != 2
                    or any(value.type != ParameterType.PARAMETER_STRING or not value.string_value.strip()
                           for value in parameters.values)):
                raise ContractError("GEOMETRY_CAPTURE_RESPONSE")
            result = copy.deepcopy(dict(scene=scene.scene, urdf=parameters.values[0].string_value,
                                        srdf=parameters.values[1].string_value))
            self._check_deadline()  # Conversion never grants a fresh budget.
            if self._closed:
                return None
            self._returned = True
            return result
        except BaseException:
            self.close()  # A secondary cleanup problem never hides the primary.
            raise

    def close(self):
        """Discard reads and destroy only owned clients; retry unconfirmed cleanup.

        Native rclpy cancelled futures have done()==False and cancel() returns
        None. Cancellation is local read bookkeeping, not a server-stop claim.
        """
        self._closed = True
        complete = True
        for read in self._reads:
            try:
                future = read["future"]
                if future is not None and not future.done() and not future.cancelled():
                    future.cancel()
                    if not future.done() and not future.cancelled():
                        complete = False
                        continue
                if read["client"] is not None:
                    if self._node.destroy_client(read["client"]) is not True:
                        complete = False
                        continue
                    read["client"] = None
                read["future"] = None
            except Exception:
                complete = False
        return complete
