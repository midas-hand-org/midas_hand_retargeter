"""Thread-safe holder for the live retargeting profile.

The web tuner writes parameters from an HTTP thread while the control loop
reads them at 60 Hz. Mutating dataclass fields in place would let the loop
observe a half-applied edit — ``curl_gain`` updated but ``curl_max_bend`` not
yet — which shows up as a visible twitch on the hand and makes "save exactly
what I am feeling right now" impossible.

So edits build a whole new frozen ``RetargetProfile`` and swap it in with a
single attribute assignment, which is atomic in CPython. Readers take one
snapshot per frame and thread it through, so the hot path needs no lock.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping

from .params import DEFAULT_PROFILE, RetargetProfile


class ProfileStore:
    """Holds the live profile, with atomic edits and bounded undo history."""

    def __init__(self, profile: RetargetProfile | None = None, *, history: int = 64):
        self._profile = profile if profile is not None else DEFAULT_PROFILE
        self._lock = threading.Lock()
        self._undo: deque[RetargetProfile] = deque(maxlen=history)
        self._redo: deque[RetargetProfile] = deque(maxlen=history)

    def get(self) -> RetargetProfile:
        """Current profile. A single attribute read, so it is never torn."""

        return self._profile

    def set(self, profile: RetargetProfile) -> RetargetProfile:
        """Replace the whole profile (e.g. loading a preset)."""

        if not isinstance(profile, RetargetProfile):
            raise TypeError(f"Expected RetargetProfile, got {type(profile).__name__}")
        with self._lock:
            self._undo.append(self._profile)
            self._redo.clear()
            self._profile = profile
            return self._profile

    def apply(self, updates: Mapping[str, object]) -> RetargetProfile:
        """Apply dotted-path ``updates`` atomically.

        Validation happens before the swap, so a bad edit leaves the live
        profile untouched rather than half-applied.
        """

        with self._lock:
            updated = self._profile.with_values(updates)
            self._undo.append(self._profile)
            self._redo.clear()
            self._profile = updated
            return updated

    def undo(self) -> RetargetProfile:
        with self._lock:
            if self._undo:
                self._redo.append(self._profile)
                self._profile = self._undo.pop()
            return self._profile

    def redo(self) -> RetargetProfile:
        with self._lock:
            if self._redo:
                self._undo.append(self._profile)
                self._profile = self._redo.pop()
            return self._profile

    def reset(self) -> RetargetProfile:
        """Return to bare defaults, keeping the change undoable."""

        return self.set(DEFAULT_PROFILE)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)
