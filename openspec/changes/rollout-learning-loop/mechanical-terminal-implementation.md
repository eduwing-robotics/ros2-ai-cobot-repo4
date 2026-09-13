# Bounded terminal software and remaining qualification

The native bounded-task caller now asks its existing motion transport for contact
context at the settled learned boundary. An available context compiles exactly the
five existing release phases from measured arm/gripper state. The terminal has its
own immutable digest and control source; the original learned plan, source rows,
grant, lease, Scene binding and absolute deadline remain unchanged. OneJob freezes
its current recorder before dispatch and retains the attempt after terminal
completion without successful-demonstration admission.

The same PickupExecutor dispatches all commands, including partial opening, hold,
full opening, retreat and safe pose. Each dispatch keeps the Scene lock and original
task/Cell checks; current state and hardware incarnation are checked. Mechanical
events remain in the existing lifecycle result under the terminal identity rather
than describing learned dataset rows. The normal diagnostic projection includes
terminal evidence and missing-contact diagnostics. Scene remains UNKNOWN when
landing is not physically observed; expected MoveIt world geometry does not assert
an observed slot, Pick success, learned Place success, or training authority.

The existing canonical diagnostic builder validates and projects mechanical
terminal evidence, unavailable-contact diagnostics and retained task handoff
metadata before computing the diagnostic digest. The live caller only persists
the unchanged lifecycle source. Web and Collection consumers reconstruct that
same diagnostic and retain strict equality/digest checks; neither consumer drops
mechanical fields to accept a response.

Native scene services retain floor/wall geometry, attach one object to
`gripper_link` with only `finger_tip_right_link` and `finger_tip_left_link` as
explicit touch links, check collision/readback, and detach at measured FK into an
expected world pose. The
[official MoveIt tutorial](https://moveit.picknik.ai/humble/doc/examples/planning_scene_ros_api/planning_scene_ros_api_tutorial.html)
supports REMOVE+ATTACH and DETACH+ADD diffs; installed Jazzy message/service types
are used by the executable fixtures. Neither mechanism proves a grasp.

## Explicitly unavailable in production

The shipped native producer reports missing qualified object/tool relation or
carried envelope, supported contact-transition/aperture model, and current required
illumination. It does not turn profile QUALIFIED strings, commanded closure,
controller completion, or policy termination into contact evidence. Its model
diagnostic derives fingertip gap from canonical robot_description; it introduces
no command inversion or independent geometry registry. A pre-start brightness
check is not continuous darkness stopping.

The pinned original URDF models increasing opening coordinate as inward fingertip
travel. The observed source equation is gap = 0.0421 - 2q meters; its configured
full-open q=0.021 gives 0.0001 m, contradicting physical opening semantics. Root
owns canonical model correction, compatibility/qualification migration, and
physical measurements. This implementation does not qualify that geometry or
claim that the real relation producer is complete. The accepted software fixture
supplies explicitly simulated contact, illumination and FK/service responses;
physical grasp, contact uncertainty, slip bounds, illumination monitoring, and
landing remain unverified.

## Executable scope

`test_native_mechanical_terminal_same_owner_stages_and_retains` exercises existing
ROS serializers/action dispatch/polling and native scene/FK/readback methods with
synthetic service clients and recorder ownership. Its rejection matrix covers
stale contact, darkness, changed source scope, scene readback mismatch, revocation
during freeze, incorrect partial-open feedback, and hardware incarnation change.
No ROS node, robot, camera, model, GPU, installation, original dataset mutation,
or live qualification is used by these fixtures.

`MechanicalDiagnosticConsumerTests` exercises the real diagnostic producer,
temporary lifecycle persistence, Web result consumption and Collection analysis
binding for unchanged, unavailable-contact, terminal and handoff results. It also
checks changed lifecycle and malformed/foreign terminal rejection. Its minimal
terminal serialization is a software consumer fixture, not physical evidence.

The diagnostic correction's focused regression changed from six failing subcases
to two passing tests. Full discovery ran 1,368 tests with three skips and two
failures in `ObjectPositionContinuityTests` (return and unknown-return scenarios,
`PHYSICAL_CONSOLE_RESOLVED_JOB`). Both failures also reproduce when the two changed
production modules are loaded from main `3811f9d` in the same environment. The
full suite is therefore recorded as FAIL, with no new failure demonstrated by
this comparison; the independent object-position issue remains outside this
diagnostic correction. These software results do not discharge physical acceptance.
