"""Suggestion helper for the Mood_detection project.

This file is optional: mood_detection.py still works if it is missing.

It does NOT chat. It asks the Gemini API for example sentences that a
person could type into the mood detector, so they know what to try.
The person still does the typing.

Setup (only needed for this feature):
    pip install google-genai
    set GEMINI_API_KEY=your_key      (free key: https://aistudio.google.com)
"""

import json
import os

try:
    from google import genai
    from google.genai import types
except ImportError:  # package not installed
    genai = None
    types = None

# Model names change over time. If this one stops working, set another one
# in cmd:  set GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
EXAMPLES_PER_MOOD = 3


def make_client():
    """Return a Gemini client, or None (after explaining why)."""
    if genai is None:
        print("Suggestions need the Gemini package. Run: pip install google-genai")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Suggestions need a Gemini API key. In cmd run: set GEMINI_API_KEY=your_key")
        return None

    return genai.Client(api_key=api_key)


def build_prompt(moods):
    """Write the instruction we send to Gemini."""
    return (
        "You help people test a mood detection program. "
        f"For each of these moods: {', '.join(moods)}. "
        f"Write {EXAMPLES_PER_MOOD} different short sentences (under 12 words each) "
        "that a person could type and that clearly show that mood. "
        "Use everyday first-person language. "
        'Answer ONLY with JSON like {"happy": ["...", "..."], "sad": ["..."]}, '
        "using exactly the mood names above as the keys."
    )


def get_suggestions(client, moods):
    """Ask Gemini and return {mood: [sentence, ...]}."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_prompt(moods),
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = json.loads(response.text)

    # Keep only the moods we asked for, and only real text.
    suggestions = {}
    for mood in moods:
        sentences = data.get(mood, [])
        suggestions[mood] = [
            s.strip() for s in sentences if isinstance(s, str) and s.strip()
        ]
    return suggestions


def show_suggestions(model, requested_mood=None):
    """Print example sentences the user can type.

    model          -- the trained mood model (we read its mood names)
    requested_mood -- optional, e.g. "sad" to get ideas for one mood only
    """
    moods = [str(m) for m in model.classes_]

    if requested_mood:
        matches = [m for m in moods if m.lower() == requested_mood.strip().lower()]
        if not matches:
            print("Unknown mood. Choose from:", ", ".join(moods))
            return
        moods = matches

    client = make_client()
    if client is None:
        return

    print("\nAsking Gemini for ideas...")
    try:
        suggestions = get_suggestions(client, moods)
    except Exception as error:  # no internet, bad key, quota used up, bad reply
        print(f"Could not get suggestions: {error}")
        return

    print("\n--- Sentences you can try typing ---")
    for mood, sentences in suggestions.items():
        print(f"\n{mood}:")
        for sentence in sentences:
            print(f"  - {sentence}")
    print("\nNow type one of them yourself (or your own sentence).")
