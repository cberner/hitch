"""Advisory file locks with explicit descriptor ownership."""

import contextlib
import fcntl
import os
from dataclasses import dataclass


@dataclass
class FileLease:
    fd: int

    def release(self) -> None:
        if self.fd < 0:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(self.fd)
        self.fd = -1


def lock_fd(fd: int, *, blocking: bool) -> FileLease | None:
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(fd, operation)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    return FileLease(fd)

