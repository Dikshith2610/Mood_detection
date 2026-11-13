
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Please add it to your .env file."
    )

client = genai.Client(api_key=api_key)


def generate_reply(message, detected_mood):

    prompt = f"""
You are an AI Mood Analysis and Text Suggestion Assistant.

The user provides a message they received from another person.
Your task is to analyze the sender's mood and suggest what the
user could text back.

Message received:
"{message}"

Mood predicted by the machine learning model:
"{detected_mood}"

Your tasks:
1. Identify the likely emotional mood of the sender.
2. Briefly explain the emotional meaning of the message.
3. Suggest exactly 3 messages the user could send to the sender.
4. Suggestions must sound natural, empathetic, and human.
5. Suggestions must be written from the user's perspective.
6. Do NOT respond directly to the original message.
7. Do NOT pretend to be the person who sent the message.
8. Avoid making unsupported assumptions about the sender.

Return only valid JSON in this format:

{{
    "mood": "emotion",
    "analysis": "short explanation",
    "suggestions": [
        "suggestion 1",
        "suggestion 2",
        "suggestion 3"
    ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        result = json.loads(response.text)

        suggestions = result.get("suggestions", [])

        if not isinstance(suggestions, list) or len(suggestions) != 3:
            raise ValueError("Gemini did not return exactly 3 suggestions.")

        return {
            "mood": result.get("mood", detected_mood),
            "analysis": result.get(
                "analysis",
                "The message may express this emotion."
            ),
            "suggestions": suggestions
        }

    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "mood": detected_mood,
            "analysis": "The message may express this emotion.",
            "suggestions": [
                "I’m here if you want to talk.",
                "Do you want to tell me more about how you feel?",
                "Let me know if there is anything I can do to help."
            ]
        }
