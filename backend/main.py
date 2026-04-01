# main.py

import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.sentiment_service import predict_emotion
from services.severity_service import predict_severity
from services.adaptive_service import run_adaptive_engine
from services.task_service import generate_task, cue_module
from services.transcription_service import transcribe_audio
from engine_inputs_outputs import EmotionState

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/test")
def test_page():
    return FileResponse("static/test.html")

# Maps the engine's integer difficulty (1–5) to the task module's string scale
DIFFICULTY_MAP = {
    1: "starter",
    2: "mild",
    3: "moderate",
    4: "severe",
    5: "severe",
}


# ---------- Request Model ----------
class TherapyRequest(BaseModel):
    text: str
    previous_difficulty: int = 3
    frustration_streak: int = 0


# ---------- Shared pipeline logic ----------
def _run_pipeline(
    text: str,
    previous_difficulty: int,
    frustration_streak: int,
    audio_file: str = None,
) -> dict:
    # 1. Sentiment — uses audio+text if audio is available, text-only otherwise
    emotion = predict_emotion(text, audio_file=audio_file)

    # 2. Severity — uses audio if available, falls back to MODERATE
    severity = predict_severity(audio_file=audio_file, text=text)

    # 3. Adaptive Engine
    decision = run_adaptive_engine(
        severity=severity,
        emotion=emotion,
        previous_difficulty=previous_difficulty,
        frustration_streak=frustration_streak,
    )

    # 4. Task generation — map int difficulty to string scale
    str_difficulty = DIFFICULTY_MAP.get(decision.difficulty, "moderate")
    task_text = generate_task(str_difficulty)

    # 5. Cue generation
    cue_sentiment = "positive" if emotion == EmotionState.POSITIVE else "negative"
    cue_output = cue_module(
        curr_task=task_text,
        curr_sentiment=cue_sentiment,
        difficulty=str_difficulty,
    )

    # 6. Update frustration streak
    updated_streak = (
        frustration_streak + 1 if emotion == EmotionState.FRUSTRATED else 0
    )

    return {
        "emotion_detected": emotion.value,
        "severity_detected": severity.value,
        "adaptive_decision": {
            "difficulty": decision.difficulty,
            "cue_type": decision.cue_type.value,
            "cue_strength": decision.cue_strength,
        },
        "task_generated": {
            "difficulty": str_difficulty,
            "prompt": task_text,
        },
        "cue": cue_output["cue"],
        "frustration_streak": updated_streak,
    }


# ---------- Text Endpoint ----------
@app.post("/therapy")
def run_therapy_pipeline(request: TherapyRequest):
    result = _run_pipeline(
        text=request.text,
        previous_difficulty=request.previous_difficulty,
        frustration_streak=request.frustration_streak,
    )
    return {"input_text": request.text, **result}


# ---------- Audio Endpoint ----------
@app.post("/therapy_audio")
async def therapy_audio(
    file: UploadFile = File(...),
    previous_difficulty: int = Form(default=3),
    frustration_streak: int = Form(default=0),
):
    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        text = transcribe_audio(file_location)
        result = _run_pipeline(
            text=text,
            previous_difficulty=previous_difficulty,
            frustration_streak=frustration_streak,
            audio_file=file_location,
        )
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

    return {"transcribed_text": text, **result}


# ---------- Health Check ----------
@app.get("/")
def health_check():
    return {"status": "Backend running"}
