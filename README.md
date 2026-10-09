# npcify

Turn the bystanders in your camera feed into cartoon avatars, live and on-device, while the people you choose stay looking like themselves.

> **Early development.** Nothing works yet; the project is being set up. Expect breaking changes.

## Quick start

You need [uv](https://docs.astral.sh/uv/) (`brew install uv` on macOS).

```bash
git clone <repo-url> npcify
cd npcify
uv sync                     # creates .venv with Python 3.11 and installs everything
uv run pytest               # run the tests
uv run python -c "import npcify"
```

## Privacy

All processing happens on your device. npcify never uploads frames and never saves faces unless you explicitly ask it to.
