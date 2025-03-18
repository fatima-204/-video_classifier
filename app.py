import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
from urllib import request
import imageio
from IPython.display import display, HTML

# Setup class names (labels)
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
    labels = [line.decode("utf-8").strip() for line in obj.readlines()]

# Define function to load I3D model
def load_i3d_model(hub_url):
    model = hub.load(hub_url).signatures['default']
    return model

# Load pre-trained model
i3d = load_i3d_model("https://tfhub.dev/deepmind/i3d-kinetics-400/1")

# Video preprocessing function
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame and normalize
        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized / 255.0
        frames.append(frame_normalized)

    # If fewer than 64 frames, pad with the first frame
    num_frames = 64
    if len(frames) < num_frames:
        frames.extend([frames[0]] * (num_frames - len(frames)))
    frames = np.array(frames)

    # Add batch dimension and channel dimension
    frames = np.expand_dims(frames, axis=0)  # Shape becomes [1, 64, 224, 224, 3]
    return frames

# Prediction function
def predict(video):
    try:
        # Preprocess video
        video_frames = preprocess_video(video)

        # Convert to tensor
        video_frames = tf.convert_to_tensor(video_frames, dtype=tf.float32)

        # Get predictions from the I3D model
        logits = i3d(video_frames)['default']
        probabilities = tf.nn.softmax(logits)

        # Get the top 5 actions
        top_5_indices = np.argsort(probabilities[0])[::-1][:5]
        top_5_predictions = []

        # Format predictions as a list of tuples (label, probability)
        for i in top_5_indices:
            if i < len(labels):  # Ensure the index is within bounds
                top_5_predictions.append((labels[i], probabilities[0][i] * 100))

        # Format the output with line breaks
        formatted_output = "<br>".join([f"{label}: {probability:.2f}%" for label, probability in top_5_predictions])

        return formatted_output

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio app setup
title = "I3D Video Classification"
description = "Classify video clips into 400 categories using an I3D model trained on Kinetics-400."

# Example list (adjust this to point to video examples you want to test)
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(
    fn=predict,  # maps input to output
    inputs=gr.Video(),  # Correct input type for video
    outputs=gr.HTML(label="Top 5 Predictions"),  # Use HTML output to support <br> tag
    examples=example_list,
    title=title,
    description=description
)

# Launch the app
demo.launch(debug=True)
