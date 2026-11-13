from sklearn.preprocessing import LabelEncoder
import pickle

# Your mood classes
moods = ["sad", "happy", "angry", "fear", "neutral"]

label_encoder = LabelEncoder()
label_encoder.fit(moods)

with open("mood_labels.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

print("mood_labels.pkl created successfully!")
print("Classes:", label_encoder.classes_)