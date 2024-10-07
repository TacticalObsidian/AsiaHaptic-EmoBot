
#!/usr/bin/env pybricks-micropython
# from pybricks.hubs import EV3Brick
# from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
#                                  InfraredSensor, UltrasonicSensor, GyroSensor)
# from pybricks.parameters import Port, Stop, Direction, Button, Color
# from pybricks.tools import wait, StopWatch, DataLog
# from pybricks.robotics import DriveBase
# from pybricks.media.ev3dev import SoundFile, ImageFile
import cv2
import numpy as np
import noisereduce as nr
from deepface import DeepFace
import math
import os
import pyttsx3
import random
import platform
import speech_recognition as sr
import pyaudio
from groq import Groq
import time


# # Create your objects here.
# ev3 = EV3Brick()

# # Write your program here.
# ev3.speaker.beep()

# Open audio cue text file
try:
    with open('audiocue.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        audio_cues = [line.strip(' \t\r\n') for line in lines if line.strip(' \t\r\n')]

    if audio_cues:  # Check if the list is not empty
         # Extract specific cues from the list
        hug_cues = audio_cues[0:3]
        quote_cues = audio_cues[3:14]
        conversational_cue = audio_cues[14:17]
        affirmative_responses = audio_cues[17:24]
        negative_responses = audio_cues[24:]
    else:
        print("No audio cues found in the file.")
except FileNotFoundError:
    print("The audiocue.txt file was not found.")

# ----------------------------EV3------------------------------------------------
def hug():
    print("Hug mechanism launched")
#----------------------------EV3------------------------------------------------

# ----------------------------CONVO-----------------------------------------------
# Initialize Groq
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Initialize libraries (TTS and STT)
speaker = pyttsx3.init()
recognizer = sr.Recognizer()

# Noise reduction function using 'noisereduce'
def reduce_noise(audio):
    data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
    reduced_noise_data = nr.reduce_noise(y=data, sr=audio.sample_rate)
    return sr.AudioData(reduced_noise_data.tobytes(), audio.sample_rate, audio.sample_width)

# Listen and process user input
def listen():
    while True:
        print("Listening...")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.energy_threshold = 300
            audio = recognizer.listen(source)
            try:
                audio = reduce_noise(audio)  # Apply noise reduction
                user_input = recognizer.recognize_google(audio)  # Convert speech to text
                return user_input
            except sr.UnknownValueError:
                speak("Sorry, I didn't quite catch that. Could you repeat?")
            except sr.RequestError as e:
                speak(f"Sorry, there was an issue with the speech service: {e}")
                return None
    # user_input = input(">>>: ")
    # return user_input

# Function to manage conversation states based on the case provided
def handle_conversation(history, user_input=None, prompt=None, case=5):
    if case == 2:  # Check if conversation should end
        history.append({"role": "user", "content": f"{user_input}. Does the user wish to end the conversation? Answer in one word, yes or no."})
    elif case == 3:  # Regular conversation continuation
        history.append({"role": "user", "content": user_input})
    elif case == 4:  # End conversation and say something encouraging
        history.append({"role": "user", "content": "The user ends the conversation. Say something encouraging and let them know you're here for them."})
    else:
        raise ValueError("Invalid case number. Use a valid case number (1-6).")

    try:
        response = client.chat.completions.create(
            messages=history,
            model="llama3-8b-8192",
            max_tokens=150,
            temperature=0.7
        ).choices[0].message.content.strip()
        
        history.append({"role": "assistant", "content": response})
    except Exception as e:
        print(f"Error: {e}")
        return None

    if case in [1, 2]:
        return response.lower() if case == 1 else (1 if response.lower() == "yes" else 0)
    return response

# Function to simulate a hug action
def hug():
    print("Hugging...")

# Text-to-speech function
def speak(message):
    speaker.say(message)
    speaker.runAndWait()
    # print(message)

# Save conversation history to file
def save_history_to_file(history):
    # Load existing history to determine the current number of lines
    existing_history = load_history_from_file()

    # If the history exceeds 60 entries, remove the first entry from the file
    if len(existing_history) >= 60:
        existing_history.pop(0)  # Remove the oldest entry

    # Write the updated history back to the file
    with open("conversation_history.txt", "w") as file:
        for entry in existing_history:
            role, content = entry["role"], entry["content"]
            file.write(f"{role}: {content}\n")

        # Append the new history entries
        for entry in history:
            role, content = entry["role"], entry["content"]
            file.write(f"{role}: {content}\n")

# Load conversation history from file
def load_history_from_file():
    history = []
    try:
        with open("conversation_history.txt", "r") as file:
            for line in file:
                line = line.strip()
                if line:  # If line not empty
                    if ": " in line:  # Check for the expected format
                        try:
                            role, content = line.split(": ", 1)
                            history.append({"role": role, "content": content.strip()})
                        except ValueError:
                            print(f"Malformed line skipped: {line}")
    except FileNotFoundError:
        print("History file not found. Starting with an empty history.")
    return history

# Initialize the conversation history
instruction = "You are a friendly assistant named EmoBot. Use simple, encouraging, and empathetic language, and prompt users to talk about their problems. Offer support if they share something serious."
history = load_history_from_file() or [{"role": "system", "content": instruction}]

def conversation():
    speak("Would you like to have a chat? Sometimes talking to someone can be relaxing.")
    user_input = listen()
    affirmative_responses = ["yes", "yup", "yeah", "y", "sure"]
    negative_responses = ["no", "nah", "no thank you"]

    # Main conversation loop
    for a in affirmative_responses:
        if a in user_input:
            convo_cue = random.choice(conversational_cue)
            speak(convo_cue)
            while True:
                user_input = listen()
                if user_input:
                    if handle_conversation(history, user_input, case=2) == 1:
                        quote_cue = random.choice(quote_cues)
                        speak(quote_cue)
                        save_history_to_file(history)
                        break
                    else:
                        response = handle_conversation(history, user_input, case=3)
                        speak(response)
                        save_history_to_file(history)
        for n in negative_responses:            
            if n in user_input:
                quote_cue = random.choice(quote_cues)
                speak(quote_cue)
                continue
    speak("Feel free to reach in for a snack!")
# ----------------------------CONVO-----------------------------------------------

# # MacOS specific initialization for pyttsx3
# if platform.system() == 'Darwin':
#     try:
#         import Foundation
#         import AppKit
#     except ImportError:
#         import objc


# Initialize video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('nodcontrol.avi', fourcc, 20.0, (640, 480))

# Distance function
def distance(x, y):
    return math.sqrt((x[0] - y[0]) * 2 + (x[1] - y[1]) * 2)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Path to face cascade
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.isfile(cascade_path):
    print(f"Error: Cascade file not found at {cascade_path}")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

# Capture source video
cap = cv2.VideoCapture(1)

# Function to get coordinates
def get_coords(p1):
    try:
        return int(p1[0][0][0]), int(p1[0][0][1])
    except:
        return int(p1[0][0]), int(p1[0][1])

# Define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX

# Define movement thresholds
gesture_threshold = 175

# Initialize face tracking
face_found = False
emotion_detected = False
while not face_found:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_center = (x + w // 2, y + h // 2)  # Initial tracking point on the face center
        face_found = True
    cv2.imshow('image', frame)
    out.write(frame)
    cv2.waitKey(1)

p0 = np.array([[face_center]], np.float32)
gesture = False
x_movement = 0
y_movement = 0
gesture_show = 60  # Number of frames a gesture is shown

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Emotion recognition
    if not emotion_detected:
        result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
        if len(result) > 0:
            emotion = result[0]['dominant_emotion']
            txt = str(emotion)
            cv2.putText(frame, txt, (50, 150), font, 1, (0, 255, 0), 3)
            cv2.imshow('image', frame)
            out.write(frame)
            cv2.waitKey(1)

            # Play a random text-to-speech cue if the emotion is sad
            if emotion == 'sad':
                hug_cue = random.choice(hug_cues)
                speaker.say(hug_cue)
                speaker.runAndWait()
                emotion_detected = True
                break  # Break the loop to start gesture detection

# cv2.destroyAllWindows()
# if we don't close the camera & open it agn seems faster ohh

# Start head gesture detection after emotion detection
if emotion_detected:
    # cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        old_gray = frame_gray.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face again to correct the drift
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_center = (x + w // 2, y + h // 2)
            p0 = np.array([[face_center]], np.float32)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None and st[0][0] == 1:
            cv2.circle(frame, get_coords(p1), 4, (0, 0, 255), -1)
            cv2.circle(frame, get_coords(p0), 4, (255, 0, 0))

            # Get the xy coordinates for points p0 and p1
            a, b = get_coords(p0), get_coords(p1)
            x_movement += abs(a[0] - b[0])
            y_movement += abs(a[1] - b[1])

            text = 'x_movement: ' + str(x_movement)
            if not gesture: cv2.putText(frame, text, (50, 50), font, 0.8, (0, 0, 255), 2)
            text = 'y_movement: ' + str(y_movement)
            if not gesture: cv2.putText(frame, text, (50, 100), font, 0.8, (0, 0, 255), 2)

            if x_movement > gesture_threshold:
                gesture = 'No'
            if y_movement > gesture_threshold:
                gesture = 'Yes'
            if gesture and gesture_show > 0:
                cv2.putText(frame, 'Gesture Detected: ' + gesture, (50, 50), font, 1.2, (0, 0, 255), 3)
                gesture_show -= 1
            if gesture_show == 0:
                if gesture == 'Yes':
                    speaker.say("I'd love to give you a comforting hug! Please place your hand near my head so I can reach out to you.")
                    speaker.runAndWait()
                    time.sleep(6)
                    conversation()
                    break
                else:
                    conversation()
                    break

        cv2.imshow('image', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    


    cv2.destroyAllWindows()
    cap.release()
    # out.release()
