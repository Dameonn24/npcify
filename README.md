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

Shows your webcam through the npcify pipeline with **every face hidden**, including yours (choosing who stays real comes in a later step).

| Key | Action |
|-----|--------|
| q / Esc | quit |
| d | show/hide debug boxes and track IDs |
| s | cycle style: pixelate-blur, pixelate, blur, solid |
| space | pause / resume |
| n | while paused, step one frame (check for leaks frame by frame) |

There is deliberately no key to turn hiding off. For development only, `--debug-no-anonymise` shows faces unhidden with a loud warning; never stream or record with it.

```bash
uv run python examples/webcam_demo.py                       # default camera, 1280x720, mirrored
uv run python examples/webcam_demo.py --video some_clip.mp4 # play a video file instead
uv run python examples/webcam_demo.py --help                # all options
```

Options: `--camera N`, `--video PATH`, `--width W`, `--height H`, `--no-mirror`, `--no-overlay`, `--style S`, `--padding W H`, `--debug`, `--det-threshold T`, `--low-threshold T`, `--det-width W`, `--max-age N`, `--recover-distance D`, `--detect-every N`.

With debug boxes on (**d**), each tracked face gets a stable ID and its own colour, with a short trail showing where it has been. Labels show the ID, state and confidence: *tentative* (just appeared), *confirmed*, or *lost* (dashed box; the detector missed it, so its position is predicted and the box grows). Thin grey boxes are the raw detections. The dots mark the eyes, nose and mouth corners.

Tuning:

- Far-away faces missed: `--det-width 1280` (slower, finds smaller faces; default 960).
- Faces you can clearly see not picked up: lower `--det-threshold` (e.g. `0.5`).
- IDs change when someone is hidden for longer (default keeps a face for 3 s): raise `--max-age` (e.g. `150`, about 5 s).
- IDs change when someone moves a lot while hidden (e.g. comes back from a spin further away): raise `--recover-distance` (e.g. `2.5`).
- A different person who appears where someone vanished takes over their ID: lower `--recover-distance` (e.g. `1.0`).
- Boxes linger in empty space after someone leaves (without walking out of frame): lower `--max-age`.

### Models

The face detector model (YuNet, from [OpenCV Zoo](https://github.com/opencv/opencv_zoo), MIT licence) is downloaded automatically the first time you run the demo, into `~/.cache/npcify/models/`, and checked against a known checksum. To download it in advance (e.g. before going offline):

```bash
uv run npcify-download-models
```

Only model files are downloaded. Video frames never leave your machine.

### Benchmark

```bash
uv run python examples/benchmark_detect.py some_clip.mp4 [--det-width 1280]
```

Prints the average detection time, the FPS that implies, and the average number of faces per frame.

To check whether tracking holds together on a clip:

```bash
uv run python examples/track_report.py some_clip.mp4 [--max-age 90]
```

It prints how many IDs were created, how long tracks lasted, and how often lost tracks were recovered. A clip with 3 people should give about 3 confirmed IDs.

### Leak check

```bash
uv run python examples/leak_check.py some_clip.mp4 [--style solid]
```

Runs the pipeline over a clip and reports detections that weren't fully covered (should always be 0), faces a second full-resolution detector can still find in the output (possible leaks, saved to `outputs/leak_check/` for review), and frames with lost tracks. Saved frames may show leaked faces: delete them when done.

Manual test checklist: [docs/testing.md](docs/testing.md).

The first time you run it on macOS you'll be asked for camera access. If the camera won't open, go to **System Settings > Privacy & Security > Camera** and allow the app you ran it from (Terminal, iTerm or VS Code), then restart that app.

## Privacy

All processing happens on your device. npcify never uploads frames and never saves faces unless you explicitly ask it to.
