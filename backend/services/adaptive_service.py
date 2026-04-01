# adaptive_service.py
# Converts raw model outputs into an EngineInput and runs the Adaptive Task Engine.
#
# CHANGE: frustration_streak is now accepted and passed into EngineInput.
# Previously it was always left at its default of 0, so the streak-based
# rule in engine_brain could never trigger.

from engine_brain import decide_next_step
from engine_inputs_outputs import EngineInput


def run_adaptive_engine(
    severity,
    emotion,
    previous_difficulty: int = 3,
    frustration_streak: int = 0
):
    """
    Converts raw model outputs into EngineInput and runs the Adaptive Task Engine.

    Args:
        severity:            SeverityLevel enum value from severity_service.
        emotion:             EmotionState enum value from sentiment_service.
        previous_difficulty: How hard the last task was (1–5). Defaults to 3.
        frustration_streak:  How many turns in a row the patient was frustrated.
                             Tracked by main.py and passed in from the request.
    """

    engine_input = EngineInput(
        severity=severity,
        emotion=emotion,
        previous_difficulty=previous_difficulty,
        frustration_streak=frustration_streak   # ADDED: was always 0 before
    )

    decision = decide_next_step(engine_input)

    return decision