import importlib

import pytest

MODULES = [
    "npcify",
    "npcify.types",
    "npcify.capture",
    "npcify.detect",
    "npcify.track",
    "npcify.decide",
    "npcify.pose",
    "npcify.render",
    "npcify.safety",
    "npcify.output",
    "npcify.pipeline",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
