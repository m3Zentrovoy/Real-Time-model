import numpy as np
import librosa
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from emotion_stream import StreamAnalyzer, AudioConfig

def run_test(filename, expected_calm_end_sec):
    path = Path(__file__).parent.parent / "data" / "audio_samples" / filename
    if not path.exists():
        path = Path(filename)
    
    if not path.exists():
        print(f"SKIP: {filename} not found")
        return False

    # Load Audio
    y, sr = librosa.load(path, sr=16000, mono=True)
    config = AudioConfig(chunk_size=0.5, buffer_size=2.0)
    bot = StreamAnalyzer(config)
    
    chunk_samples = int(config.sr * config.chunk_size)
    
    scores = []
    timestamps = []
    
    print(f"Testing {filename}...")
    
    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]
        if len(chunk) < chunk_samples: break # Skip partial end
        
        res = bot.process_chunk(chunk)
        scores.append(res['agitation_score'])
        timestamps.append(res['timestamp'])

    # Analysis
    scores = np.array(scores)
    timestamps = np.array(timestamps)
    
    # 1. Check Initial Calm (0 to expected_calm_end_sec)
    # We expect the AVG score in this region to be < 60
    calm_mask = timestamps < expected_calm_end_sec
    if np.sum(calm_mask) > 0:
        calm_scores = scores[calm_mask]
        avg_calm = np.mean(calm_scores)
        max_calm = np.max(calm_scores)
    else:
        avg_calm = 0
        max_calm = 0
        
    # 2. Check Agitation (After calm zone)
    # We expect significant portion to be > 60
    active_mask = timestamps > expected_calm_end_sec
    if np.sum(active_mask) > 0:
        active_scores = scores[active_mask]
        avg_active = np.mean(active_scores)
    else:
        avg_active = 0
    
    print(f"  [0-{expected_calm_end_sec}s] Calm Zone: Avg={avg_calm:.1f}, Max={max_calm:.1f}")
    print(f"  [>{expected_calm_end_sec}s] Active Zone: Avg={avg_active:.1f}")
    
    # ASSERTIONS
    passed = True
    
    # Criterion 1: Calm Start
    if avg_calm > 55: # Tolerance threshold (below 60)
        print("  FAILED: Start was too loud/agitated.")
        passed = False
    
    # Criterion 2: Transition Detected
    if avg_active < 65: 
        print("  FAILED: Did not detect enough agitation later.")
        passed = False
        
    if passed:
        print("  PASSED ✅")
    else:
        print("  FAILED ❌")
        
    return passed

if __name__ == "__main__":
    print("--- AUTOMATED MODEL VERIFICATION ---")
    # china_angry_discuss: First 9s should be CALM
    p1 = run_test("china_angry_discuss.wav", expected_calm_end_sec=9.0)
    
    if p1:
        print("\nOVERALL STATUS: READY FOR PRODUCTION")
        exit(0)
    else:
        print("\nOVERALL STATUS: NEEDS TUNING")
        exit(1)
