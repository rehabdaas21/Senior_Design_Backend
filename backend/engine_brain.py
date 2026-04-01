# engine_brain.py
# This file contains the "brain" of the Adaptive Task Engine (ATE).
# It reads the patient's current state (inputs) and returns a decision (outputs).

# Import the shared Input/Output definitions
from engine_inputs_outputs import EngineInput, EngineDecision, SeverityLevel, EmotionState, CueType


def clamp(value: int, low: int, high: int) -> int:
    # Keeps a number within a safe range (e.g., difficulty always stays 1..5)
    return max(low, min(high, value))


def decide_next_step(engine_input: EngineInput) -> EngineDecision:
    """
    This function is the single entry point to the Adaptive Task Engine.
    It receives the patient's current state and returns a decision.

    Rules are evaluated top-to-bottom. The first matching rule wins.
    Priority order: SEVERE → frustration_streak → FRUSTRATED → MODERATE → POSITIVE
    """

    # -----------------------------------------------------------------------
    # Rule 1 (Emergency Break):
    # SEVERE severity always takes top priority regardless of emotion.
    # The patient needs maximum protection — drop difficulty and suggest a break.
    # -----------------------------------------------------------------------
    if engine_input.severity == SeverityLevel.SEVERE:
        new_difficulty = clamp(engine_input.previous_difficulty - 2, 1, 5)

        return EngineDecision(
            difficulty=new_difficulty,
            cue_type=CueType.BREAK,
            cue_strength=2
        )

    # -----------------------------------------------------------------------
    # Rule 2 (Frustration Streak):
    # If the patient has been frustrated 3+ times in a row, suggest a break
    # even if the current emotion has recovered slightly.
    # This uses frustration_streak which is now properly tracked and passed in.
    # -----------------------------------------------------------------------
    if engine_input.frustration_streak >= 3:
        new_difficulty = clamp(engine_input.previous_difficulty - 1, 1, 5)

        return EngineDecision(
            difficulty=new_difficulty,
            cue_type=CueType.BREAK,
            cue_strength=2
        )

    # -----------------------------------------------------------------------
    # Rule 3 (Protect — Frustrated):
    # If the patient is frustrated, ease difficulty and offer a hint to help.
    # SLOW_DOWN is used here instead of ENCOURAGEMENT — it gives the patient
    # space to breathe rather than just cheering them on.
    # -----------------------------------------------------------------------
    if engine_input.emotion == EmotionState.FRUSTRATED:
        new_difficulty = clamp(engine_input.previous_difficulty - 1, 1, 5)

        return EngineDecision(
            difficulty=new_difficulty,
            cue_type=CueType.SLOW_DOWN,
            cue_strength=2
        )

    # -----------------------------------------------------------------------
    # Rule 4 (Moderate Support):
    # MODERATE severity with a positive emotion — hold difficulty steady
    # but offer a hint to keep the patient supported without pushing them.
    # -----------------------------------------------------------------------
    if (
        engine_input.emotion == EmotionState.POSITIVE
        and engine_input.severity == SeverityLevel.MODERATE
    ):
        return EngineDecision(
            difficulty=engine_input.previous_difficulty,
            cue_type=CueType.HINT,
            cue_strength=1
        )

    # -----------------------------------------------------------------------
    # Rule 5 (Challenge):
    # POSITIVE emotion + MILD severity → patient is doing well.
    # Gently increase difficulty and give light encouragement.
    # -----------------------------------------------------------------------
    if (
        engine_input.emotion == EmotionState.POSITIVE
        and engine_input.severity == SeverityLevel.MILD
    ):
        new_difficulty = clamp(engine_input.previous_difficulty + 1, 1, 5)

        return EngineDecision(
            difficulty=new_difficulty,
            cue_type=CueType.ENCOURAGEMENT,
            cue_strength=1
        )

    # -----------------------------------------------------------------------
    # Default (Steady):
    # No rule matched — hold steady with light encouragement.
    # -----------------------------------------------------------------------
    return EngineDecision(
        difficulty=engine_input.previous_difficulty,
        cue_type=CueType.ENCOURAGEMENT,
        cue_strength=1
    )