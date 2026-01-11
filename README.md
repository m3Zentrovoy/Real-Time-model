# Real-Time Emotion (Insight Genie)

Real-time emotional state estimation (Agitation/Arousal) system based on audio.  
Includes basic acoustic analysis and an optional text layer (ASR + Sentiment).

## Project Structure

```text
Real-Time-model/
├── src/                    # Source code (analysis algorithms)
│   ├── emotion_stream.py   # Main class StreamAnalyzer
│   ├── mood_mapping.py     # Interpretation logic (CALM/NOT_CALM)
│   └── arousal_features.py # Feature extraction
├── notebooks/              # Usage examples (Jupyter)
│   ├── adaptive_realtime_emotion.ipynb # Interactive demonstration
│   └── verification_demo.ipynb         # Test run on data
├── apps/
│   └── streamlit_demo/     # Web interface prototype (Streamlit)
├── data/
│   └── audio_samples/      # Audio sample files
└── tests/
    └── auto_verify.py      # Automated verification script
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Demonstration (Jupyter)**:
   Open `notebooks/adaptive_realtime_emotion.ipynb` and follow the steps inside.
   
3. **Run Streamlit Dashboard**:
   ```bash
   cd apps/streamlit_demo
   streamlit run dashboard.py
   ```

## How It Works

The system accepts an audio stream (in chunks of 0.5-1 sec), extracts acoustic features (loudness, pitch, voice jitter), and calculates an Agitation Score (0-100).

- **Calm**: Score < 60
- **Not Calm (Agitation/Aggression)**: Score > 60

Additionally, the system attempts to classify frustration indicators and tone (Loud, High-Pitch, Trembling).

## Verification

To run automated tests:
```bash
python3 tests/auto_verify.py
```
The script will verify the model's operation on reference files (e.g., `china_angry_discuss.wav` from `data/audio_samples`).
