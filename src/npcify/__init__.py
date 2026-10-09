"""npcify: replace bystanders' faces with cartoon avatars in live video, on-device."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npcify.pipeline import Pipeline

__all__ = ["Pipeline"]
__version__ = "0.0.1"


def __getattr__(name: str) -> Any:
    # Imported lazily so `import npcify` and `python -m npcify.<module>` stay light.
    if name == "Pipeline":
        from npcify.pipeline import Pipeline

        return Pipeline
    raise AttributeError(f"module 'npcify' has no attribute {name!r}")
