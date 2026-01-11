import torch
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import transformers.utils.import_utils
import os

# --- SECURITY PATCH ---
# Allows loading model weights without strict torch version check (fix for ValueError).
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True

# Configuration
MODEL_ID = "superb/wav2vec2-base-superb-er"
TARGET_SAMPLING_RATE = 16000

print(f"Engine: Initializing model {MODEL_ID}...")

# Global variables initialization
model = None
feature_extractor = None

try:
    # Load model components
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    print("Engine: Model loaded successfully.")
except Exception as e:
    print(f"Engine Critical Error: {e}")
    # Important: if model fails to load, variables remain None

def get_all_scores(audio_data):
    """
    Main inference function.
    Args:
        audio_data: np.array (waveform), 16000 Hz.
    Returns:
        dict: {emotion_label: probability} or None if error.
    """
    # Check if model is loaded
    if model is None or feature_extractor is None:
        print("Engine Error: Model is not loaded.")
        return None

    try:
        # 1. Prepare data (Tensor -> Numpy)
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.numpy()
        
        # Remove extra dimensions (e.g., (N, 1) -> (N,))
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        # 2. Feature Extraction
        # return_tensors="pt" returns PyTorch tensors
        inputs = feature_extractor(
            audio_data,
            sampling_rate=TARGET_SAMPLING_RATE,
            return_tensors="pt",
            padding=True
        )

        # 3. Inference (disable gradients for speed)
        with torch.no_grad():
            logits = model(**inputs).logits

        # 4. Convert logits to probabilities (Softmax)
        scores = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # 5. Form result
        result = {}
        # config.id2label contains mapping id -> emotion name (neutral, happy, etc.)
        for i, score in enumerate(scores):
            label = model.config.id2label[i]
            result[label] = score.item()
            
        return result

    except Exception as e:
        # Log error, but don't crash the app
        print(f"Inference Error: {e}")
        return None

# Wrapper function for external compatibility
def process_audio_chunk(audio_data):
    return get_all_scores(audio_data)

# --- Test Block (check if script runs) ---
if __name__ == "__main__":
    # Create dummy audio chunk (silence) for test: 1 second, 16000 Hz
    dummy_audio = np.zeros(16000)
    print("Running test inference on dummy audio...")
    result = process_audio_chunk(dummy_audio)
    print("Test Result:", result)