import numpy as np

def map_mood_tags(analysis_result: dict, history: list) -> dict:
    """
    Maps the streaming analysis result to the 9 specific business tags.
    """
    feats = analysis_result.get("features", {})
    score = analysis_result.get("agitation_score", 0.0)
    state = analysis_result.get("state", "CALM")
    
    rms = feats.get("rms", 0.0)
    pitch = feats.get("pitch_mean", 0.0)
    jitter = feats.get("pitch_jitter", 0.0)
    
    # 1. Sentiment Volatility (Variance of score over last 5 entries)
    # ----------------------------------------------------------------
    last_5_scores = [h['agitation_score'] for h in history[-5:]] if history else [score]
    volatility = float(np.std(last_5_scores)) if len(last_5_scores) > 1 else 0.0
    
    # 2. Tone Description
    # ----------------------------------------------------------------
    tones = []
    if rms > 0.05: tones.append("Loud")
    if pitch > 200: tones.append("High-Pitch")
    if jitter > 0.03: tones.append("Trembling")
    if not tones: tones.append("Neutral")
    tone_str = ", ".join(tones)
    
    # 3. Frustration Indicators
    # ----------------------------------------------------------------
    # High Jitter + High Energy often means stress/frustration
    frustration_level = (jitter * 500) + (rms * 100)
    is_frustrated = frustration_level > 20.0
    
    return {
        "sentiment_volatility": round(volatility, 2),
        "emotional_displays": "Aggression" if score > 80 else "None",
        "emotional_state_alignment": "Aligned" if score < 40 else "Misaligned",
        "frustration_indicators": "Detected" if is_frustrated else "None",
        "frustration_control": "Low" if is_frustrated and score > 70 else "High",
        "tone_description": tone_str,
        "sentiment_trend": "Rising" if score > (last_5_scores[0] if last_5_scores else 0) else "Stable",
        "polarity_change_frequency": 0, # Placeholder for complex logic
        "emotional_swing_intensity": round(score / 10.0, 1)
    }
