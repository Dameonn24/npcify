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

## Running the demo

Shows your webcam through the npcify pipeline with an FPS overlay. Press **q** or **Esc** to quit.

```bash
uv run python examples/webcam_demo.py                       # default camera, 1280x720, mirrored
uv run python examples/webcam_demo.py --video some_clip.mp4 # play a video file instead
uv run python examples/webcam_demo.py --help                # all options
```

Options: `--camera N`, `--video PATH`, `--width W`, `--height H`, `--no-mirror`, `--no-overlay`.

The first time you run it on macOS you'll be asked for camera access. If the camera won't open, go to **System Settings > Privacy & Security > Camera** and allow the app you ran it from (Terminal, iTerm or VS Code), then restart that app.

## Privacy

All processing happens on your device. npcify never uploads frames and never saves faces unless you explicitly ask it to.
