# npcify — project rules

## What this is

npcify is an open-source Python library that takes a live camera feed or a video and replaces bystanders' faces with cartoon (Wii-style) avatars, while chosen people, such as the creator, keep looking like themselves. The avatars follow each person's head angle and expression. The near-term goal is a live pipeline at ~30 FPS that can feed a virtual camera. Longer term the aim is live bystander avatarising everywhere, eventually in AR, so people can film in public without exposing the people around them.

## Core rules (non-negotiable)

1. **Privacy by default.** All processing stays on-device. Never upload frames anywhere. Never save faces, face crops or faceprints (embeddings) to disk unless the user explicitly opts in. No telemetry.
2. **Fail safe.** If the system is unsure about a face (low detection confidence, lost track, unknown identity, avatar can't be rendered), it must blur it. A real bystander face must never slip through. When in doubt, hide it.

## Target hardware

Apple Silicon Macs first (developed on a MacBook Air M5, no NVIDIA GPU). Prefer libraries and backends that run well on CPU / Apple GPU (Metal, Core ML). Other platforms come later.

## Code style

- Small, single-purpose modules (see `src/npcify/`).
- Type hints everywhere. Shared data shapes live in `src/npcify/types.py`.
- Every module has a test in `tests/`.
- Use ruff for linting and formatting: `uv run ruff check .` and `uv run ruff format .`.
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
2. Webcam in/out with FPS counter
3. Face detection
4. Tracking
5. Blur everyone
6. Keep list with faceprints
7. Pose and expressions
8. 2D avatars
9. Safety net and delay buffer
10. Virtual camera and video file output
11. Speed pass
12. Clean library API
13. Open-source launch kit
