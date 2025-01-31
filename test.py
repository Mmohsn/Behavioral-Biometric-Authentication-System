import pandas as pd
import numpy as np
import time
import keyboard
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
DURATION = 10  # Duration for keystroke data collection
KEY_HOLD = 'hold_time'
KEY_RELEASE = 'release_time'
TYPING_SPEED = 'typing_speed'
def collect_keystroke_data(prompt):
    print(prompt)
    keystrokes = []
    start_time = time.time()
    while time.time() - start_time < DURATION:
        event = keyboard.read_event(suppress=True)
        timestamp = time.time()
        keystrokes.append((event, timestamp))
    return keystrokes

def extract_features(keystrokes):
    features = []
    for i in range(1, len(keystrokes)):
        prev_event, prev_timestamp = keystrokes[i - 1]
        event, timestamp = keystrokes[i]

        if event.event_type == keyboard.KEY_DOWN and prev_event.event_type == keyboard.KEY_DOWN:
            hold_time = timestamp - prev_timestamp
            release_time = 0  # Not applicable for key down events
            typing_speed = 0  # Not applicable for key down events
        elif event.event_type == keyboard.KEY_UP and prev_event.event_type == keyboard.KEY_DOWN:
            hold_time = 0  # Not applicable for key up events
            release_time = timestamp - prev_timestamp
            typing_speed = 1 / (timestamp - prev_timestamp)
        else:
            continue

        features.append({
            KEY_HOLD: hold_time,
            KEY_RELEASE: release_time,
            TYPING_SPEED: typing_speed
        })

    return features



def load_trained_model(model_path, encoder_path):
    # Load the trained neural network model
    model = load_model(model_path)

    # Load the label encoder
    with open(encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)

    return model, label_encoder
def make_prediction(model, label_encoder, df):
    # Assuming `new_data` is a DataFrame with the same structure as the training data minus the label column
    # You would need to perform any scaling or preprocessing that was done during the initial model training
    # Make predictions
    predictions_prob = model.predict(df)
    predictions_index = np.argmax(predictions_prob, axis=1)
    predictions_labels = label_encoder.inverse_transform(predictions_index)

    # Optionally, return the class probabilities as well
    return predictions_labels, predictions_prob
    
print("Please log in to your account.")
username_keystrokes = collect_keystroke_data("Collecting data. Please type your password:")
username_features = extract_features(username_keystrokes)
data = {'Features': username_features}
df = pd.DataFrame(data['Features'])  

model, label_encoder = load_trained_model('model.keras', 'label_encoder.pkl')
predictions, probabilities = make_prediction(model, label_encoder, df)
unique_classes, counts = np.unique(predictions, return_counts=True)
total_predictions = counts.sum()
max_index = counts.argmax()
max_class = unique_classes[max_index]
print(f"Welcome {max_class}")

"""
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Please log in to your account.")
username_keystrokes = collect_keystroke_data("Collecting data. Please type your password:")
username_features = extract_features(username_keystrokes)
data = {'Features': username_features}
df = pd.DataFrame(data['Features'])
# Flatten and predict
predicted_classes = model.predict(df)

# Count occurrences and calculate percentages
unique_classes, counts = np.unique(predicted_classes, return_counts=True)
total_predictions = counts.sum()

# Find the class with the maximum count
max_index = counts.argmax()
max_class = unique_classes[max_index]
max_percentage = counts[max_index] / total_predictions * 100
if max_class == 'unauthourized':
    print("login Field")
print(f"Login Success Welcome {max_class}")
"""