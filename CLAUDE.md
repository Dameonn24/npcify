# npcify — project rules

## What this is

npcify is an open-source Python library that takes a live camera feed or a video and replaces bystanders' faces with cartoon (Wii-style) avatars, while chosen people, such as the creator, keep looking like themselves. The avatars follow each person's head angle and expression. The near-term goal is a live pipeline at ~30 FPS that can feed a virtual camera. Longer term the aim is live bystander avatarising everywhere, eventually in AR, so people can film in public without exposing the people around them.

## Core rules (non-negotiable)

1. **Privacy by default.** All processing stays on-device. Never upload frames anywhere. Never save faces, face crops or faceprints (embeddings) to disk unless the user explicitly opts in. No telemetry.
2. **Fail safe.** If the system is unsure about a face (low detection confidence, lost track, unknown identity, avatar can't be rendered), it must blur it. A real bystander face must never slip through. When in doubt, hide it.
3. **Cover every track and every detection.** Tentative and lost tracks are still real faces. Any step that hides faces must cover every non-removed track (using its box, which for lost tracks is a predicted, enlarged box) *and* every raw detection, not just confirmed tracks. Confirmation only matters for things like stable avatar assignment, never for deciding whether to hide someone.

## Known limitations (by design)

- A person who leaves and comes back much later (after `max_age` frames, ~2 s) gets a new track ID. Recognising returning people is step 6 (keep list with faceprints).
- The reverse can also happen: a lost track can be recovered by a *different* person who appears near where it vanished (within `recover_distance` face-widths), so they inherit its ID. A track ID is therefore never proof of identity. Step 6 must re-verify "keep real" with the faceprint after any recovery (lost → confirmed), and never let a bystander inherit the creator's keep-real status through a track ID.

## Target hardware

Apple Silicon Macs first (developed on a MacBook Air M5, no NVIDIA GPU). Prefer libraries and backends that run well on CPU / Apple GPU (Metal, Core ML). Other platforms come later.

## Code style

- Small, single-purpose modules (see `src/npcify/`).
- Type hints everywhere. Shared data shapes live in `src/npcify/types.py`.
- Every module has a test in `tests/`.
- Use ruff for linting and formatting: `uv run ruff check .` and `uv run ruff format .`.
- Development-only visuals (boxes, landmarks) live in `debug.py`, never in the real output path.
- Model weights are never committed. Register them in `src/npcify/models.py` with a URL and SHA-256; they download to `~/.cache/npcify/models/` on first use.
- Never commit photos or videos of real people. Local test images go in `tests/assets/local/` (git-ignored); tests that need them must skip when absent.
- Manual test checklist: `docs/testing.md`.
- Python 3.11, managed with uv. Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`), only when a step needs them.

## Commands

```bash
uv sync                   # install / update the environment
uv run pytest             # tests
uv run ruff check .       # lint
uv run ruff format .      # format
```

## Build plan

1. Setup (project skeleton, tooling, tests) ✅
2. Webcam in/out with FPS counter ✅
3. Face detection ✅
4. Tracking ✅
5. Blur everyone
6. Keep list with faceprints
7. Pose and expressions
8. 2D avatars
9. Safety net and delay buffer
10. Virtual camera and video file output
11. Speed pass
12. Clean library API
13. Open-source launch kit
