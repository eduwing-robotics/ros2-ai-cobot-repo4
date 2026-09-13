# Native dependency patches

`frcobot_ros2.patch` applies to the project's pinned ROS driver submodule.
Normal setup continues to use that patch and the existing SDK binary.

`frcobot_ros2-overlap-candidate.patch` is an **unqualified diagnostic delta**, not
part of normal setup or permission to execute learned actions. It removes the
project's gripper-completion ARM barrier and post-gripper servo restart while
retaining fault/source checks, the existing worker and completion evidence.
It does not enable arbitrary in-flight gripper target supersession.

The reviewed baseline `fairino_hardware_interface.cpp` SHA-256 is
`fa25f5755cc73d5b57e96c5b27e352cd2bc7fd228150848ef4b5a85c14862843`;
the candidate is `eea7624a5c5cedf5356d9a7317f8acae26c1ebb4e175967f13797d6516b5c003`.
Apply only to an isolated copy of that v5 baseline, never the active install.
The existing native fixture's `overlap_write` scenario distinguishes baseline
ServoJ call counts `0/0` from candidate `1/2` with fresh synthetic v5 telemetry;
both must stop further calls on an injected UDP fault. This is SDK-call routing,
not physical motion or coexistence qualification. A single-package build passed
with the selected coherent SDK; deployment, native ARM/gripper overlap and
qualified exact-plan test consumption remain open. No previously qualified
“2-degree” exercise is assumed without its original evidence.

Source review limits this candidate to one gripper command in an isolated
exercise. Removing the pause exposes a worker handoff interval between taking
the pending target and publishing RPC/active-generation state; repeated target
updates are not qualified. The current `check_hardware(allow_pending=True)` also
rejects an active gripper generation with `arm_resumed=1`. Under the candidate,
that flag describes an unpaused ARM producer, not a successful post-gripper
`ServoMoveStart` acknowledgement. Do not falsify either field to satisfy the old
consumer. Normal integration must separate these responsibilities first.

`fairino-cpp-sdk-2.3.7.patch` is an **unqualified, opt-in SDK candidate** against
[FAIRINO's v2.3.7-3.9.7 source](https://github.com/FAIR-INNOVATION/fairino-cpp-sdk/tree/0553c35d760a4e76c9b8d2fc0208ca83e6d731cd).
It reuses the CNDE receiver and adds a coherent, nonblocking snapshot getter.
Receipt time and local publication identity do not prove acquisition age,
command completion or physical safety. Legacy getters are not made coherent.
Active receive configuration changes return BUSY; stopped changes invalidate
the snapshot. Native FR5 consumption and physical qualification are separate.

The reproducible opt-in test exports the pinned commit from a local SDK clone,
applies the patch in temporary storage and builds/tests it without installation
or device calls. It verifies actual library selection and zero network syscalls:

```sh
FR5_FAIRINO_SDK_REPO=/path/to/fairino-cpp-sdk direnv exec . \
  python3 -m unittest tests.data_factory.rollout.test_sdk_snapshot -v
```

No download or SDK build is added to ordinary unit discovery. Upstream SDK source
and patch context are covered by [Apache-2.0](fairino-sdk-LICENSE); the patch
identifies the modified files. No manufacturer binary or dataset is included.
