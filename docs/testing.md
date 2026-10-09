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
| 3 | Someone 3–5 m away | One person stands at the far side of the room. | Detected most of the time at 3 m. At 4–5 m some flicker is possible at the default `--det-width 960`; try `--det-width 1280` (slower), which should hold them more steadily. |
| 4 | Side profile | Turn your head slowly from facing the camera to a full side-on view. | Detected up to roughly 45–60°. Full profile (90°) may drop out; note the angle where it does. |
| 5 | Partly covered face | Hand over the mouth, a mug, a face mask, sunglasses, hair over one eye. | Still detected with any one of those, perhaps at a lower score (0.6–0.8). Hand covering most of the face may drop out; that's acceptable for now. |
| 6 | Low light | Turn off the main lights and use one lamp or the screen glow. | Still detected, maybe at a lower score. FPS may drop to 15–25 because macOS lengthens the camera exposure; that's the camera, not npcify (check that `detect` stays a few ms). |
| 7 | Fast movement | Move your head quickly side to side, then walk across the frame. | Box follows with at most an occasional missed frame during motion blur. No boxes left behind in empty space. |

Also check:

- **Nothing on blank scenes.** Point the camera at a wall or a busy bookshelf: no boxes on non-faces (occasional low-score false positives on posters or photos of faces are expected, since those *are* faces).
- **q / Esc quits cleanly** and the camera light goes off.

## Tracking (step 4)

Watch the `#N` label on each face and the confirmed / tentative / lost counts in the overlay.

| # | Scenario | How to test | Good enough looks like |
|---|----------|-------------|------------------------|
| 1 | Moving around | Lean, sway, and move around the frame for 30 s. | One ID the whole time. `tentative` only for the first ~3 frames. |
| 2 | Head turn / spin | Turn to full profile and back, then spin all the way round on your chair (under 3 s). | Same ID after turning back. A dashed *lost* box while you're side-on is fine; it should stay over your head. |
| 3 | Hands over face | Cover your face with both hands for 2–3 s, then remove them. | Same ID. Dashed box stays over the face while covered. |
| 4 | Two people crossing | Two people walk past each other, one in front of the other. | Each keeps their own ID most of the time. An occasional swap when both faces fully overlap is acceptable. |
| 5 | Leave and return | Walk out of frame, wait 3 s, come back. | New ID on return (expected; recognising people is step 6). If you walk out past the frame edge the old box disappears within a few frames. If you leave without crossing the edge (e.g. duck below the desk), it stays for up to 3 s; that's deliberate. |
| 6 | False positives | Point the camera at a busy scene with no people. | Any brief false box stays *tentative* and vanishes; never *confirmed*. |

For a recorded clip, `uv run python examples/track_report.py clip.mp4` should show about one confirmed ID per person.

## Hiding faces (step 5)

Run `uv run python examples/webcam_demo.py`. Use **space** to pause and **n** to step one frame at a time through each scenario. Every face must be covered in **every** frame. One visible frame is a failure.

| # | Scenario | How to test | Good enough looks like |
|---|----------|-------------|------------------------|
| 1 | Still | Sit still facing the camera. | Face, hair line, ears and chin all inside the mosaic. Background untouched. |
| 2 | Quick head turns | Turn left/right quickly, look down and up. | Covered in every frame, including mid-turn and full profile. |
| 3 | Hand waves | Wave a hand across your face quickly, several times. | No frame where any part of your eyes/nose/mouth is visible. |
| 4 | Walk in / out | Walk in from the side of the frame and out again. | Covered from the first frame your face is visible (partly visible at the edge counts). |
| 5 | Far away | Stand 3–5 m back. | Covered. If not, note the distance: that's a detector miss (try `--det-width 1280`). |
| 6 | Several people | 2–3 people, moving, crossing. | Everyone covered every frame. |
| 7 | Styles | Press **s** to cycle; also try `--style solid`. | Each style hides the face. FPS stays 25+. |

Press **d** to see why something is covered: lost tracks (dashed) should still sit over the face. A frame where a face shows is a bug: note what was happening, and if you can, record a clip and run `examples/leak_check.py` on it.

## Keep list (step 6)

Turn on debug (**d**) to see each face's status (`hide 0.21`, `keep: person 1 0.68`, `enrolling …`).

| # | Scenario | How to test | Good enough looks like |
|---|----------|-------------|------------------------|
| 1 | Enrol | Click your face, look at the camera, move your head slightly. | Enrolment finishes and you're uncovered within ~1 s of clicking. |
| 2 | Leave and return | Walk out of frame, wait 5 s, come back. | Covered when you reappear, uncovered again within ~0.5 s. |
| 3 | Head turns | Turn side-on and back. | You may blur briefly when found again after being lost; never stay blurred more than ~0.5 s once facing the camera. |
| 4 | Someone else | A second person (ideally similar-looking) sits in frame and moves around. | Always covered. Their debug score should stay well below the threshold (0.40). |
| 5 | Swap seats | Two people cross or swap places, one kept. | The other person is never uncovered (a frame or two near the crossing would be a bug to report). |
| 6 | Remove | Right-click your face, then press **c**. | You're covered again straight away. |
| 7 | 3–4 people | Several people in frame, one kept. | FPS stays 25+. |

## Pose and expressions (step 7)

Needs a second person (bystander): your own face is kept real only if you click it, so without a click you can test on yourself. Turn on debug (**d**).

| # | Scenario | Good enough looks like |
|---|----------|------------------------|
| 1 | Turn left/right | `y` goes + when the face turns toward the right of the window, - toward the left, in both normal and `--no-mirror` mode. Blue axis points where the nose points. |
| 2 | Nod up/down | `p` goes + looking up, - looking down. |
| 3 | Tilt head | `r` goes + tilting clockwise as seen in the window. |
| 4 | Blink, wink | `blinkL`/`blinkR` jump for the eye on that side of the window, within a frame or two. Winks show on one side only. |
| 5 | Mouth, smile, brows | `mouth`, `smile`, `brow+` (surprised) and `brow-` (frown) bars respond. |
| 6 | Hold still | Numbers settle; they shouldn't wander by more than a degree or two. |
| 7 | Move back 3–5 m | Source switches from `lm` to `coarse`; yaw and roll still roughly right, bars hidden. |
| 8 | 3–4 bystanders | FPS stays 25+. |

## Avatars (step 8)

| # | Scenario | Good enough looks like |
|---|----------|------------------------|
| 1 | Bystander appears | Blurred first, then a cartoon head fades in within ~0.3 s. |
| 2 | Turn, nod, tilt | The head turns (features slide, more hair shows), nods and tilts with them, without lag or wobble. |
| 3 | Blink, talk, smile, brows | Eyes close, mouth opens and closes while talking, mouth curves when smiling, brows move. |
| 4 | Step through (space, n) | No frame shows any part of a real face, including during the fade-in and when a track is lost. |
| 5 | Lost / far / new faces | Dropped back to blur instantly, never a frozen avatar over an empty space. |
| 6 | Kept person | Stays real, never gets an avatar. |
| 7 | Keys | **a** toggles avatars/blur, **r** gives new looks, **p** saves a screenshot. |
| 8 | 3–4 bystanders | FPS 25+. |

## Record results

Note the date, the lighting, the settings used (`--det-width`, `--det-threshold`) and which scenarios failed. Failures in 3–7 are inputs to the safety-net work in step 9, where anything uncertain must be blurred rather than shown.

## Optional: local face images for automated tests

Put your own `.jpg`/`.png` photos with clearly visible faces in `tests/assets/local/`. `uv run pytest` will then also check that a face is found in each one. That folder is git-ignored; never commit photos of real people.
