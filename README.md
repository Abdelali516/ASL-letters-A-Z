# Real-Time Sign Language Recognition with Local LLM Interaction

This project is a real-time sign language recognition system that translates hand gestures into text and allows interaction with a local AI model.

Using a webcam, the system detects hand landmarks, predicts the corresponding sign language letter using a trained neural network, and gradually builds a sentence in real time. When the user removes their hand from the camera for a short period, the completed sentence is automatically sent to a locally running AI model for processing and response generation.

## Project Overview

The goal of this project is to explore the integration of computer vision, machine learning, and AI interaction in a single real-time system. The application recognizes sign language letters, converts them into text, and uses a local language model to interpret or respond to the generated sentence.

## Dataset

The dataset used for training the model was **created from scratch for this project**.
Hand landmark data was collected using MediaPipe, processed, and labeled to build a dataset representing different sign language letters.

## Neural Network

The neural network used for classification was **designed, trained, and tested as part of this project**.

The model takes hand landmark coordinates as input and predicts the corresponding letter of the alphabet. Training and evaluation were performed using Python machine learning tools to ensure the model performs reliably in real-time conditions.

The model achieved an **average classification accuracy of approximately 97%** on the test data.

## AI Integration

The AI component of the system was also implemented as part of this project. The application connects to a **locally running language model using Ollama**, specifically the **Llama 3.1 8B model**, to process the generated sentence.

Running the model locally allows the system to function **without relying on external APIs or cloud services**.

## Features

* Real-time hand detection and tracking using MediaPipe
* Custom dataset creation for sign language letters
* Neural network built, trained, and tested for gesture classification
* Average model accuracy of **97%**
* Time-based stabilization to reduce noisy predictions
* Real-time sentence generation from recognized letters
* Automatic sentence submission after hand absence
* Local AI interaction using Ollama and the Llama 3.1 8B model

## Special Cases

Some sign language letters involve motion rather than static hand shapes.

* **J** and **Z** require movement in standard sign language.
  In the current implementation, these letters are represented using **constant hand signals as placeholders**.

Future updates will include proper motion-based gesture recognition for these letters.

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* Scikit-learn
* Ollama
* Llama 3.1 8B

## Future Improvements

* Implement motion-based recognition for dynamic letters such as **J** and **Z**
* Improve prediction stability using temporal voting methods
* Add a graphical interface to display the detected sentence
* Expand the dataset and improve model accuracy
* Support additional sign language gestures beyond the alphabet

## Purpose

This project demonstrates how computer vision, machine learning, and local AI models can be combined to build an interactive system capable of translating sign language gestures into meaningful text and AI-driven responses.
