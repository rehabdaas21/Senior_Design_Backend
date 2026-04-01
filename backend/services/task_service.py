# openAI for task gen and sounddevice + scipy for audio gen
# on the terminal/bash use this: python -m pip install openai sounddevice scipy (to install this --> pip install openai sounddevice scipy)

## V. IMP note: the audio doesn't work on colab only on VS code or on the python IDLE !!!!!

import random
import os
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI

# task bank --> if the API fails
Tasks = {
  "starter": [
      "Can you tell me about your day?"
  ],

  "mild": [
      "Can you tell me about something you did this morning?",
      "Can you tell me about a time you spent with your family?",
      "Can you tell me about a simple happy memory?",
      "Can you retell a story you know well in a simple way?"
  ],

  "moderate": [
      "Can you tell me about a family experience that you remember well?",
      "Can you tell me about a familiar event from your life and what happened first, next, and last?",
      "Can you tell me about a memory that made you feel proud?",
      "Can you retell the story of Cinderella in your own words?"
  ],

  "severe": [
      "Can you tell me a personal story from beginning to end with as much detail as you can?",
      "Can you describe an important life event and explain what happened step by step?",
      "Can you retell a story you know well and include the main events in order?",
      "Can you tell me about a meaningful memory and explain why it was important to you?"
  ]
}

task_categories = {
  "starter": ["day"],
  "mild": ["memory", "family_event"],
  "moderate": ["memory", "family_event", "story_retelling"],
  "severe": ["memory", "family_event", "story_retelling"]
}

# Set your key with: setx OPENAI_API_KEY "your-key-here"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=10.0)

sampleR = 16000
dur = 10  # in secs --> we can increase it later


def generate_task(difficulty):
  """
  This function uses OpenAI's API key to generate tasks based on the difficulty that the ATE outputs.
  If the API fails then it gets the tasks from the Task bank that has a couple of fixed prompts.
  """
  try:
    task_type = random.choice(task_categories[difficulty])
    prompt = f"""
    Generate one storytelling therapy task for an aphasia rehabilitation patient.

    Difficulty level: {difficulty}
    Task type: {task_type}

    Return only the task itself. Do not add labels, explanations, bullet points, or extra text.

    Rules:
    - There are 4 levels: starter, mild, moderate, severe.
    - The task must always be storytelling-based and open-ended.
    - Do not generate picture description tasks.
    - Keep the wording supportive, simple, and easy to understand.
    - Make the task different each time.
    - Encourage the patient to speak in full thoughts.
    - Use only these storytelling styles:
      1. personal narratives about daily life
      2. memories
      3. family experiences or familiar events
      4. retelling a known story
    - Do not generate the same task. Always change it evven if the difficulty doesn't change.

    Task type rules:
    - day: ask them how they are, or how was their day.
    - memory: ask about a personal memory.
    - family_event: ask about a family experience or familiar event.
    - story_retelling: ask the patient to retell a known story.

    Difficulty guidelines:
    - starter: always ask the patient about their day in a short, warm, everyday way.
    - mild: ask for a simple personal narrative or a simple retelling of a familiar story.
    - moderate: ask for more detail, a clearer sequence of events, a memory, or a family/familiar event.
    - severe: ask for a fuller and more detailed story, possibly with a beginning, middle, and end, or a known story retelling with the main events in order.

    Generate exactly one task for the given difficulty level.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        timeout=10
    )
    return response.choices[0].message.content.strip()

  except Exception:
    return random.choice(Tasks[difficulty])

def audioGen(filename="output.wav", duration=dur, sample_rate=sampleR):
  """
  This function records the audio from the patient's microphone and saves it as .wav file.
  """
  print("\nRecording...")
  audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
  sd.wait()
  write(filename, sample_rate, audio)
  print("\nRecording ended.")
  print("\nSaved at:", os.path.abspath(filename))
  return filename

def task_module(difficulty="starter", round_no=1):
  """
  This function generates the task based on the difficulty that the ATE outputs to the task module.
  It returns the task, the difficulty, and the generated audio file as a .wav file.

  Inputs: difficulty from the ATE
  Outputs: a dict with the task, difficulty, and the audio file (output.wav)
  """
  task = generate_task(difficulty)

  print("\nTask:", task)
  input("\nPress enter to begin recording.")
  filename = f"round_{round_no}.wav"
  audio_file = audioGen(filename=filename)
  return {
      "task": task,
      "difficulty": difficulty,
      "audio_file": audio_file
  }


########## CUE MODULE HERE ###########

# CUE bank --> backup if the API fails
cue_bank = {
    "positive": [
        "You're doing well! Keep going and tell a little more.",
        "Nice job. Try to explain your story in a bit more detail."
    ],
    "negative": [
        "That's okay! Take your time. Think about the first part of the story.",
        "You're doing fine. Start with one simple detail, and we can build from there."
    ]
}

def generate_cues(curr_sentiment, curr_task, difficulty):
  """
  Generates one supportive cue for the patient based on sentiment, current task, and difficulty.
  Falls back to a fixed cue bank if the API fails.
  """
  try:
      prompt = f"""
      Generate one short supportive cue for an aphasia rehabilitation patient.

      Difficulty level: {difficulty}
      Sentiment level: {curr_sentiment}
      Current task: {curr_task}

      Rules:
      - There are 4 difficulty levels: starter, mild, moderate, severe.
      - The patient is already able to see the task, so do NOT repeat or restate or rephrase the full task.
      - Do NOT paraphrase the full task.
      - The cue must help the patient answer the task by giving one small hint.
      - The hint should suggest one starting point, such as:
        - the beginning of the story
        - one person involved
        - one place
        - one event
        - one feeling or memory
      - Give one small helpful hint that supports the patient in answering.
      - Also include one short encouragement sentence.
      - Keep the wording supportive, simple, and easy to understand.
      - The cue must be related to the current task, but should not sound like the task itself.
      - Include a short encouragement message.
      - There are 2 sentiment levels: positive and negative.
      - If sentiment is negative, give a stronger hint and you may suggest taking a short break.
      - If sentiment is positive, first encourage the patient and then gently help them continue but don't ask them anything and don't talk about the task.
      - Keep the response short: 1 to 2 sentences only.
      - Return only the cue text and nothing else.
      """

      response = client.chat.completions.create(
          model="gpt-4.1-mini",
          messages=[{"role": "user", "content": prompt}],
          timeout=10
      )
      return response.choices[0].message.content.strip()

  except Exception:
      return random.choice(cue_bank[curr_sentiment])

def cue_module(curr_task, curr_sentiment, difficulty="starter"):
  """
  This function generates the cues based on the sentiment, current task and difficulty.
  It returns the cue, the sentiment, the difficulty, and the task.
  """
  cue = generate_cues(curr_sentiment, curr_task, difficulty)

  return {
      "cue": cue,
      "difficulty": difficulty,
      "task": curr_task,
      "sentiment": curr_sentiment
  }
