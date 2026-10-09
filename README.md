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

Options: `--camera N`, `--video PATH`, `--width W`, `--height H`, `--no-mirror`, `--no-overlay`, `--no-show-detections`, `--det-threshold T`, `--low-threshold T`, `--det-width W`, `--max-age N`, `--detect-every N`.

Each tracked face gets a stable ID and its own colour, with a short trail showing where it has been. Labels show the ID, state and confidence: *tentative* (just appeared), *confirmed*, or *lost* (dashed box; the detector missed it, so its position is predicted and the box grows). Thin grey boxes are the raw detections. The dots mark the eyes, nose and mouth corners.

Tuning:

- Far-away faces missed: `--det-width 960` (slower, finds smaller faces).
- Faces you can clearly see not picked up: lower `--det-threshold` (e.g. `0.5`).
- IDs change when someone turns their head or is covered for a moment: raise `--max-age` (e.g. `30`, about 1 second) or lower `--low-threshold` (e.g. `0.2`).
- Boxes linger in empty space after someone leaves: lower `--max-age`.

### Models

The face detector model (YuNet, from [OpenCV Zoo](https://github.com/opencv/opencv_zoo), MIT licence) is downloaded automatically the first time you run the demo, into `~/.cache/npcify/models/`, and checked against a known checksum. To download it in advance (e.g. before going offline):

```bash
uv run npcify-download-models
```

Only model files are downloaded. Video frames never leave your machine.

### Benchmark

```bash
uv run python examples/benchmark_detect.py some_clip.mp4 [--det-width 960]
```

Prints the average detection time, the FPS that implies, and the average number of faces per frame.

To check whether tracking holds together on a clip:

```bash
uv run python examples/track_report.py some_clip.mp4 [--max-age 30]
```

It prints how many IDs were created, how long tracks lasted, and how often lost tracks were recovered. A clip with 3 people should give about 3 confirmed IDs.

Manual test checklist: [docs/testing.md](docs/testing.md).

The first time you run it on macOS you'll be asked for camera access. If the camera won't open, go to **System Settings > Privacy & Security > Camera** and allow the app you ran it from (Terminal, iTerm or VS Code), then restart that app.

## Privacy

All processing happens on your device. npcify never uploads frames and never saves faces unless you explicitly ask it to.
