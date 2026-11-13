import pickle

with open("mood_labels.pkl", "rb") as file:
    label_encoder = pickle.load(file)

print(type(label_encoder))
print(label_encoder)