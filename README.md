# Real-Time Emotion (Insight Genie)

Short pipeline to estimate agitation from audio in near real-time.
Includes a main acoustic score and an optional text layer (ASR + sentiment).

## Quick start
- Install deps: `pip install -r requirements.txt` (or Step 0 in the notebook).
- Open `adaptive_realtime_emotion.ipynb`.
- Run steps 1 → 2 → 3 (others are optional).

## Workflow
1. Step 1: pick a file from `audio_samples/` and click **Load file**.
2. Step 2: add GT labels (CALM / NOT_CALM) in `GT_PRESETS`.
3. Step 3: see the main plot **Agitation vs Ground Truth**.
4. Step 3b: metrics on a 2s grid.
5. Step 3c: feature ranking (optional).
6. Step 3d: clean vs noise comparison (optional).
7. Step 4: per-feature plots (optional).

## Noise pairs
- Noisy files must end with `_with_noise` or `_with_noice`.
- Pairs are auto-built: `test.wav` ↔ `test_with_noise.wav`.
- GT uses the base key without the noise suffix.

## Structure
- `audio_samples/` — input audio (wav/mp3).
- `adaptive_realtime_emotion.ipynb` — main pipeline.
- `arousal_features.py` — standalone feature extraction (CLI).

## Notes
- Text layer (ASR + sentiment) is optional and may require internet on first model download.
- Noise can shift the score strongly — use Step 3d to validate robustness.
