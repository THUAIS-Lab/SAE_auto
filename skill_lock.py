"""Cross-thread and cross-process locking for a shared skills directory."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import ContextManager, Dict, Iterator, Tuple

import fcntl


_LOCK_SH = getattr(fcntl, "LOCK_SH", 1)
_LOCK_EX = getattr(fcntl, "LOCK_EX", 2)
_LOCK_NB = getattr(fcntl, "LOCK_NB", 4)
_LOCK_UN = getattr(fcntl, "LOCK_UN", 8)

# key -> (mode, depth). Context-local state makes exclusive maintenance calls
# re-entrant when they reuse case helpers that normally take the shared lock.
_HELD_SKILL_LOCKS: ContextVar[Dict[str, Tuple[str, int]]] = ContextVar(
    "held_skill_locks",
    default={},
)


def _lock_path(skills_dir: Path) -> Path:
    return Path(skills_dir) / ".maintenance.lock"


@contextmanager
def skill_directory_lock(
    skills_dir: Path,
    *,
    exclusive: bool,
    blocking: bool = True,
) -> Iterator[bool]:
    """Acquire the shared skills lock and yield whether it was acquired."""
    directory = Path(skills_dir).resolve()
    key = str(directory)
    held = dict(_HELD_SKILL_LOCKS.get())
    existing = held.get(key)
    requested_mode = "exclusive" if exclusive else "shared"

    if existing is not None:
        mode, depth = existing
        if mode == "shared" and exclusive:
            raise RuntimeError("cannot upgrade a shared skills lock to exclusive")
        held[key] = (mode, depth + 1)
        token = _HELD_SKILL_LOCKS.set(held)
        try:
            yield True
        finally:
            _HELD_SKILL_LOCKS.reset(token)
        return

    directory.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(directory)
    handle = lock_path.open("a+", encoding="utf-8")
    operation = _LOCK_EX if exclusive else _LOCK_SH
    if not blocking:
        operation |= _LOCK_NB
    try:
        try:
            fcntl.flock(handle, operation)
        except BlockingIOError:
            yield False
            return
        held[key] = (requested_mode, 1)
        token = _HELD_SKILL_LOCKS.set(held)
        try:
            yield True
        finally:
            _HELD_SKILL_LOCKS.reset(token)
            fcntl.flock(handle, _LOCK_UN)
    finally:
        handle.close()


def shared_skill_lock(skills_dir: Path) -> ContextManager[bool]:
    return skill_directory_lock(skills_dir, exclusive=False, blocking=True)


def exclusive_skill_lock(skills_dir: Path, *, blocking: bool = True) -> ContextManager[bool]:
    return skill_directory_lock(skills_dir, exclusive=True, blocking=blocking)
