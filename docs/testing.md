# Manual testing checklist

Automated tests can't check a real camera or real faces, so run through this by hand after changes to capture or detection.

```bash
uv run python examples/webcam_demo.py
```

Watch the boxes, the confidence scores, the `faces:` count and the `detect` time in the overlay. "Flicker" means a box disappearing for a frame or two while the face is still visible.

## Face detection (step 3)

| # | Scenario | How to test | Good enough looks like |
|---|----------|-------------|------------------------|
| 1 | One face, close up | Sit 40–80 cm from the laptop, facing the camera. | Box on your face every frame, score 0.85+, 5 dots on eyes, nose and mouth corners. Overall FPS 25+. |
| 2 | Several people | 2–4 people in frame at once, at different distances. | `faces:` matches the number of people. No duplicate boxes on one face. FPS stays 25+. |
| 3 | Someone 3–5 m away | One person stands at the far side of the room. | Detected most of the time at 3 m. At 4–5 m some flicker is expected at the default `--det-width 640`; try `--det-width 960`, which should hold them more steadily. |
| 4 | Side profile | Turn your head slowly from facing the camera to a full side-on view. | Detected up to roughly 45–60°. Full profile (90°) may drop out; note the angle where it does. |
| 5 | Partly covered face | Hand over the mouth, a mug, a face mask, sunglasses, hair over one eye. | Still detected with any one of those, perhaps at a lower score (0.6–0.8). Hand covering most of the face may drop out; that's acceptable for now. |
| 6 | Low light | Turn off the main lights and use one lamp or the screen glow. | Still detected, maybe at a lower score. FPS may drop to 15–25 because macOS lengthens the camera exposure; that's the camera, not npcify (check that `detect` stays a few ms). |
| 7 | Fast movement | Move your head quickly side to side, then walk across the frame. | Box follows with at most an occasional missed frame during motion blur. No boxes left behind in empty space. |

Also check:

- **Nothing on blank scenes.** Point the camera at a wall or a busy bookshelf: no boxes on non-faces (occasional low-score false positives on posters or photos of faces are expected, since those *are* faces).
- **q / Esc quits cleanly** and the camera light goes off.

## Record results

Note the date, the lighting, the settings used (`--det-width`, `--det-threshold`) and which scenarios failed. Failures in 3–7 are inputs to the safety-net work in step 9, where anything uncertain must be blurred rather than shown.

## Optional: local face images for automated tests

Put your own `.jpg`/`.png` photos with clearly visible faces in `tests/assets/local/`. `uv run pytest` will then also check that a face is found in each one. That folder is git-ignored; never commit photos of real people.
