#!/usr/bin/env python3
"""
Simple arousal pipeline for audio:
- energy (RMS)
- pitch (YIN): median, IQR, jitter
- voiced_ratio (ratio of voiced frames)

Usage:
  python arousal_features.py audio.wav --window 2.0 --hop 0.5 --out out.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import soundfile as sf
import librosa


@dataclass
class WindowFeatures:
    start_sec: float
    end_sec: float
    rms: float
    pitch_median: float
    pitch_iqr: float
    pitch_jitter: float
    voiced_ratio: float


def compute_features(
    audio: np.ndarray, sr: int, window_sec: float, hop_sec: float
) -> List[WindowFeatures]:
    """Computes arousal features using sliding windows."""
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    if win <= 0 or hop <= 0:
        raise ValueError("window_sec and hop_sec must be > 0")

    results: List[WindowFeatures] = []
    n = len(audio)

    for start in range(0, n, hop):
        end = start + win
        if start >= n:
            break
        seg = audio[start:end]
        if seg.size == 0:
            continue

        rms = float(np.sqrt(np.mean(np.square(seg), dtype=np.float64)))
        if rms < 1e-6:
            results.append(
                WindowFeatures(
                    start_sec=start / sr,
                    end_sec=min(end, n) / sr,
                    rms=rms,
                    pitch_median=0.0,
                    pitch_iqr=0.0,
                    pitch_jitter=0.0,
                    voiced_ratio=0.0,
                )
            )
            if end >= n:
                break
            continue

        # Pitch track inside window (YIN). Reduce frame_length for short windows.
        frame_length = min(2048, max(512, len(seg)))
        hop_length = max(128, frame_length // 4)
        fmin, fmax = 50.0, 400.0
        try:
            pitch_track = librosa.yin(
                seg,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            pitch_track = pitch_track[np.isfinite(pitch_track)]
        except Exception:
            pitch_track = np.array([], dtype=np.float32)

        if pitch_track.size:
            pitch_median = float(np.median(pitch_track))
            q25, q75 = np.quantile(pitch_track, [0.25, 0.75])
            pitch_iqr = float(q75 - q25)
            if pitch_track.size > 1:
                diffs = np.abs(np.diff(pitch_track))
                base = np.maximum(pitch_track[:-1], 1e-6)
                pitch_jitter = float(np.mean(diffs / base))
            else:
                pitch_jitter = 0.0
            voiced_ratio = float(np.count_nonzero(pitch_track) / len(pitch_track))
        else:
            pitch_median = 0.0
            pitch_iqr = 0.0
            pitch_jitter = 0.0
            voiced_ratio = 0.0

        results.append(
            WindowFeatures(
                start_sec=start / sr,
                end_sec=min(end, n) / sr,
                rms=rms,
                pitch_median=pitch_median,
                pitch_iqr=pitch_iqr,
                pitch_jitter=pitch_jitter,
                voiced_ratio=voiced_ratio,
            )
        )

        if end >= n:
            break

    return results


def load_audio(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Reads audio, converts to mono and target_sr, normalizes to [-1,1]."""
    audio, sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 1.0:
        audio = audio / max_abs
    return audio, sr


def save_csv(rows: Iterable[WindowFeatures], out_path: Path) -> None:
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(out_path, index=False)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Computes arousal features (energy/pitch/voiced_ratio) by windows."
    )
    parser.add_argument("audio_path", help="Path to WAV/audio file")
    parser.add_argument("--window", type=float, default=2.0, help="Window length, sec (2.0)")
    parser.add_argument("--hop", type=float, default=0.5, help="Window hop, sec (0.5)")
    parser.add_argument("--out", type=str, default=None, help="Path to CSV (if not specified - print to stdout)")
    args = parser.parse_args(argv)

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise SystemExit(f"File not found: {audio_path}")

    audio, sr = load_audio(audio_path)
    feats = compute_features(audio, sr, window_sec=args.window, hop_sec=args.hop)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_csv(feats, out_path)
        print(f"Saved: {out_path} ({len(feats)} windows)")
    else:
        df = pd.DataFrame([asdict(r) for r in feats])
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
