"""Decide who stays real and who becomes an avatar.

People on the keep list (e.g. the creator) stay real; everyone else is
avatarised. When in doubt, the answer is "not real".
"""

from __future__ import annotations

from npcify.types import Decision, Frame, Track


def decide(frame: Frame, tracks: list[Track]) -> list[Decision]:
    """Return one keep-real-or-not decision per track."""
    raise NotImplementedError
