import numpy as np
import librosa
from collections import deque
from dataclasses import dataclass, field

@dataclass
class AudioConfig:
    sr: int = 16000
    chunk_size: float = 0.5  # Seconds
    buffer_size: float = 2.0 # Seconds (Context window for pitch/rms)
    hop_length: int = 512

@dataclass
class StreamState:
    buffer: deque = field(default_factory=lambda: deque(maxlen=32000)) # 2s @ 16k
    last_score: float = 0.0
    state_streak: int = 0
    current_state: str = "CALM"
    processed_count: int = 0
    history: list = field(default_factory=list)

class UniversalBaseline:
    """
    Hardcoded 'Normal Human' stats to solve Start-of-Call screaming.
    """
    def __init__(self):
        # These values are empirical approximations of "Normal Speech"
        # These values are empirical approximations of "Normal Speech"
        # RELAXED SETTINGS to avoid False Positives on loud but normal speech
        self.stats = {
            'rms_median': 0.06,       # Was 0.08 (Tightened slightly)
            'rms_mad': 0.03,          # Was 0.04
            'pitch_mean_median': 150.0, 
            'pitch_mean_mad': 50.0,
            'pitch_jitter_median': 0.03, # Was 0.04 (Tightened slightly)
            'pitch_jitter_mad': 0.015    # Was 0.02
        }

class StreamAnalyzer:
    def __init__(self, config: AudioConfig = AudioConfig()):
        self.config = config
        self.state = StreamState()
        # Resize buffer based on actual SR
        maxlen = int(config.sr * config.buffer_size)
        self.state.buffer = deque(maxlen=maxlen)
        self.baseline = UniversalBaseline()

    def process_chunk(self, chunk: np.ndarray) -> dict:
        """
        Ingest a small chunk (e.g. 0.5s), update buffer, and return metrics.
        """
        # 1. Update Buffer
        self.state.buffer.extend(chunk)
        
        # 2. Check if we have enough data (at least 0.1s)
        if len(self.state.buffer) < int(self.config.sr * 0.1):
            return self._empty_result()

        audio_window = np.array(self.state.buffer)
        
        # 3. Extract Features (Fast)
        feats = self._extract_features(audio_window)
        
        # 4. Compute Agitation Score (vs Universal Baseline)
        score = self._compute_score(feats)
        
        # 5. Smooth & Update State
        final_score, state = self._update_state_logic(score)
        
        timestamp = self.state.processed_count * self.config.chunk_size
        self.state.processed_count += 1
        
        return {
            "timestamp": timestamp,
            "agitation_score": final_score,
            "raw_score": score,
            "state": state,
            "features": feats
        }

    def _extract_features(self, y: np.ndarray) -> dict:
        rms = float(np.sqrt(np.mean(y**2)))
        
        # Pitch (Fast method using librosa.pyin is too slow for real-time, 
        # but for simulation we use it. In prod, use YIN C++ or simpler ZCR/Autocorr proxy)
        # For <2s latency, pyin on 2s buffer is acceptable (~0.1s on CPU).
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=self.config.sr, frame_length=1024)
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) > 0:
            pitch_mean = np.mean(valid_f0)
            pitch_std = np.std(valid_f0)
            if len(valid_f0) > 2:
                jitter = np.mean(np.abs(np.diff(valid_f0))) / (pitch_mean + 1e-6)
            else:
                jitter = 0.0
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            jitter = 0.0

        return {
            "rms": rms,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_jitter": jitter
        }

    def _compute_score(self, feats: dict) -> float:
        # Weighted simple Z-score logic
        b = self.baseline.stats
        
        def z(val, name):
            return max(0, (val - b[f'{name}_median']) / (b[f'{name}_mad'] * 2))

        score = 0.0
        score += z(feats['rms'], 'rms') * 35.0         # Was 30.0 (Boosted)
        score += z(feats['pitch_mean'], 'pitch_mean') * 20.0
        score += z(feats['pitch_jitter'], 'pitch_jitter') * 20.0
        
        return min(100.0, score)

    def _update_state_logic(self, raw_score: float) -> tuple[float, str]:
        # Simple smoothing
        alpha = 0.3
        smoothed = alpha * raw_score + (1 - alpha) * self.state.last_score
        self.state.last_score = smoothed
        
        # State Hysteresis
        if smoothed > 60:
            new_state = "NOT_CALM"
        else:
            new_state = "CALM"
            
        return round(smoothed, 1), new_state

    def _empty_result(self):
        return {"agitation_score": 0.0, "state": "CALM", "features": {}}
