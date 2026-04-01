#This file defines the inputs and outputs of the Adaptive Task Engine.

#Inputs describe the patient's current state(severity, emotion, etc)
#Outputs describe the system's decision (task type, difficult, and support)

#Keeping these definitions in one place helps the system stay clear,
#consistent, and easy to make sense of.


# We use Enum to define values that are limited to specific choices
#(for example: emotion can ONLY be positive or frustrated)
from enum import Enum

# We use dataclass to group related pieces of data together 
#(for example: keeping all the patient state information needed by the Adaptive Task Engine in one place)
from dataclasses import dataclass 

#This enum defines the ONLY emotion states the system is allowed to use
#It prevents random or misspelled emotion values
class EmotionState(Enum):
    POSITIVE="positive" # patient sounds engaged or confident 
    FRUSTRATED="frustrated" #patient sounds stressed or discouraged 

#This enum deines the ONLY severity levels the system is allowed to use 
# It represents how impaired the patient's speech currently is 
class SeverityLevel(Enum):
    MILD="mild" # small difficulty, patient mostly fluent 
    MODERATE="moderate" # noticeable difficulty, needs support 
    SEVERE="severe"# strong difficulty, needs protection and simplicity 


class CueType(Enum):
    ENCOURAGEMENT="encouragement" # emotional support (e.g.,"you're doing great!")
    HINT="hint"                 # subtle help related to the task  (e.g.,"try thinking about a happy memory")
    SLOW_DOWN="slow_down"       # ask the patient to take their time (e.g.,"take your time")
    BREAK="break"               # suggest a pause (e.g.,"let's take a short break")


#This dataclass groups all the information about the patient 
# that the Adaptive Task Engine needs to make a decision 
@dataclass
class EngineInput: 
    severity: SeverityLevel # how severe the aphasia is
    emotion: EmotionState  # how the patient is currently feeling
    previous_difficulty: int=3 # how hard the last task was(1=easy, 5=hard)
    frustration_streak: int=0  # how many times the patient was frustrated in a row 

# This dataclass represents the decision made by the Adaptive Task Engine
#It answers the question: "What should the system do next?"
@dataclass
class EngineDecision: 
    difficulty: int #how hard the next task should be(1=easy, 5=hard)
    cue_type: CueType #how the system should support the patient 
    cue_strength: int # how strong the cue is(0=none, 1=light, 2=strong)
