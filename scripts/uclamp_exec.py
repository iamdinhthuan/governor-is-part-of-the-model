"""Set this task's utilization clamp (uclamp.min) and exec a command.

Usage: python3 uclamp_exec.py <uclamp_min 0..1024> -- <command ...>
The clamp is inherited across exec and by threads created afterwards, so
schedutil sees at least this utilization whenever the task is runnable.
"""

import ctypes
import os
import sys

SYS_SCHED_SETATTR = 274  # aarch64
SCHED_FLAG_KEEP_POLICY = 0x08
SCHED_FLAG_KEEP_PARAMS = 0x10
SCHED_FLAG_UTIL_CLAMP_MIN = 0x20


class SchedAttr(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("sched_policy", ctypes.c_uint32),
                ("sched_flags", ctypes.c_uint64), ("sched_nice", ctypes.c_int32),
                ("sched_priority", ctypes.c_uint32), ("sched_runtime", ctypes.c_uint64),
                ("sched_deadline", ctypes.c_uint64), ("sched_period", ctypes.c_uint64),
                ("sched_util_min", ctypes.c_uint32), ("sched_util_max", ctypes.c_uint32)]


def main() -> None:
    value = int(sys.argv[1])
    command = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    attr = SchedAttr(size=ctypes.sizeof(SchedAttr),
                     sched_flags=SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_KEEP_PARAMS
                     | SCHED_FLAG_UTIL_CLAMP_MIN,
                     sched_util_min=value)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(SYS_SCHED_SETATTR, 0, ctypes.byref(attr), 0) != 0:
        raise OSError(ctypes.get_errno(), "sched_setattr(uclamp_min) failed")
    if command:
        os.execvp(command[0], command)
    print("uclamp_min set to", value)


if __name__ == "__main__":
    main()
