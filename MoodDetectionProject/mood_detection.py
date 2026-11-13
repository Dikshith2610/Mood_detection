
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from chatbot import generate_reply


# -----------------------------------
# Load trained LSTM model
# -----------------------------------

MODEL_PATH = "mood_lstm.keras"
TOKENIZER_PATH = "mood_tokenizer.pkl"
LABELS_PATH = "mood_labels.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

with open(LABELS_PATH, "rb") as file:
    label_encoder = pickle.load(file)


# -----------------------------------
# Text preprocessing
# -----------------------------------

MAX_LEN = 100


def predict_mood(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_index = np.argmax(prediction, axis=1)[0]

    predicted_mood = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(prediction[0][predicted_index])

    return predicted_mood, confidence


# -----------------------------------
# Main interactive console
# -----------------------------------

print("=" * 50)
print("       MOOD DETECTION & TEXT SUGGESTION")
print("=" * 50)

print("Enter a message you received from someone.")
print("Type 'exit' to stop the program.")


while True:

    text = input("\nEnter the message: ").strip()

    if text.lower() == "exit":
        print("Exiting Mood Detection System...")
        break

    if not text:
        print("Please enter a valid message.")
        continue

    try:

        # Step 1: Predict mood using LSTM
        detected_mood, confidence = predict_mood(text)

        print("\n--- LSTM Mood Prediction ---")
        print("Detected Mood:", detected_mood)
        print(f"Model Confidence: {confidence:.2%}")

        # Step 2: Analyze mood and generate texting suggestions
        print("\nAnalyzing the message and generating suggestions...")

        chat_result = generate_reply(
            text,
            detected_mood
        )

        # Step 3: Display Gemini analysis
        print("\n--- Gemini Mood Analysis ---")
        print("Mood:", chat_result["mood"])
        print("Analysis:", chat_result["analysis"])

        # Step 4: Display suggestions
        print("\n--- Suggested Texts You Can Send ---")

        for index, suggestion in enumerate(
            chat_result["suggestions"],
            start=1
        ):
            print(f"{index}. {suggestion}")

    except Exception as error:
        print("\nAn error occurred:")
        print(error)

    print("\n" + "-" * 50)
