# main.py
# MVP: анализ эмоций каждые 0.5 сек и отправка индикаторов по WebSocket

import asyncio
from collections import deque
from typing import Dict

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# ========= ПРОСТЫЙ АНАЛИЗ ГОЛОСА (ФЕЙК, БЕЗ LIBROSA) =========

def analyze_voice_chunk(audio_bytes: bytes) -> Dict[str, float]:
    """
    Простейший анализ 500 ms аудио.
    Здесь нет реального pitch/MFCC, только пример:
    - чем больше абсолютных значений, тем "громче" и "злее"
    """
    if not audio_bytes:
        return {"loudness": 0.0}

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return {"loudness": 0.0}

    # Нормализуем в -1..1
    audio = audio / 32768.0
    loudness = float(np.mean(np.abs(audio)))
    return {"loudness": loudness}


def detect_emotion(features: Dict[str, float]) -> Dict[str, float]:
    """
    Простейшая rule-based эмоция по громкости.
    Чем громче – тем злее. Это просто демонстрация пайплайна.
    """
    loudness = features.get("loudness", 0.0)

    emotions = {
        "angry": 0.0,
        "frustrated": 0.0,
        "sad": 0.0,
        "calm": 0.0,
        "neutral": 0.0,
    }

    if loudness > 0.4:
        emotions["angry"] = 0.7
        emotions["frustrated"] = 0.3
    elif loudness > 0.2:
        emotions["frustrated"] = 0.6
        emotions["neutral"] = 0.4
    elif loudness > 0.1:
        emotions["neutral"] = 0.7
        emotions["calm"] = 0.3
    else:
        emotions["calm"] = 0.9

    total = sum(emotions.values()) or 1.0
    emotions = {k: v / total for k, v in emotions.items()}
    return emotions


# ========= РАСЧЕТ ИНДИКАТОРОВ =========

class IndicatorCalculator:
    def __init__(self):
        self.emotion_history = deque(maxlen=60)  # ~30 сек при шаге 0.5 c
        self.prev_dominant = None
        self.polarity_changes = 0
        self.elapsed = 0.0

    def step(self, emotions: Dict[str, float]) -> Dict[str, object]:
        """Вызывается каждые 0.5 сек, возвращает индикаторы."""
        self.elapsed += 0.5
        self.emotion_history.append(emotions)

        dominant = max(emotions, key=emotions.get)
        conf = emotions[dominant]

        # смена доминирующей эмоции
        if self.prev_dominant and self.prev_dominant != dominant:
            self.polarity_changes += 1
        self.prev_dominant = dominant

        # volatility: сколько раз менялась эмоция за последние 20 шагов
        recent = list(self.emotion_history)[-20:]
        dom_seq = [max(e, key=e.get) for e in recent] if recent else []
        changes = sum(
            1 for i in range(1, len(dom_seq)) if dom_seq[i] != dom_seq[i - 1]
        )
        sentiment_volatility = min(10.0, changes / 3.0)

        # frustration_level: проста комбинация angry + frustrated
        frustration_level = (
            emotions.get("angry", 0.0) * 0.6
            + emotions.get("frustrated", 0.0) * 0.4
        ) * 100.0

        # trend: растет / падает / стабилен по интенсивности
        trend = "INITIALIZING"
        if len(self.emotion_history) >= 10:
            last5 = list(self.emotion_history)[-5:]
            prev5 = list(self.emotion_history)[-10:-5]
            last_int = np.mean([max(e.values()) for e in last5])
            prev_int = np.mean([max(e.values()) for e in prev5])
            if last_int > prev_int * 1.2:
                trend = "ESCALATING"
            elif last_int < prev_int * 0.8:
                trend = "DE_ESCALATING"
            else:
                trend = "STABLE"

        indicators = {
            "dominant_emotion": dominant,
            "confidence": conf,
            "sentiment_volatility": sentiment_volatility,
            "frustration_level": frustration_level,
            "polarity_change_frequency": self.polarity_changes,
            "trend": trend,
            "time_description": f"{int(self.elapsed)} seconds into call",
            "emotional_displays": [
                e for e, v in emotions.items() if v > 0.15
            ],
            "emotions": emotions,
        }
        return indicators


# ========= FASTAPI + WEBSOCKET =========

app = FastAPI(title="Emotion Indicators MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/{call_id}")
async def ws_call(websocket: WebSocket, call_id: str):
    """
    Ожидает бинарные чанки аудио (16‑bit PCM),
    каждые ~0.5 сек считает эмоции и индикаторы и шлет JSON.
    """
    await websocket.accept()
    calc = IndicatorCalculator()

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            features = analyze_voice_chunk(audio_bytes)
            emotions = detect_emotion(features)
            indicators = calc.step(emotions)

            await websocket.send_json(
                {
                    "call_id": call_id,
                    "indicators": indicators,
                }
            )

    except Exception:
        await websocket.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # запуск: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
