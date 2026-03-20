# ASL Real-Time Sign Language Recognition with Local LLM Interaction

A real-time American Sign Language (ASL) recognition system that translates hand gestures into text and enables interaction with a locally running AI model — no cloud APIs required.

---

## Overview

This project integrates **computer vision**, **deep learning**, and **local AI** into a single real-time pipeline. Using a webcam, the system:

1. Detects hand landmarks in real time using MediaPipe
2. Predicts the corresponding ASL letter using a custom-trained neural network
3. Builds a sentence progressively as letters are recognized
4. Automatically submits the sentence to a local LLM (Llama 3.1 8B via Ollama) when the hand leaves the frame

---

## Features

- 🖐️ Real-time hand detection and tracking using MediaPipe
- 🧠 Custom FFNN trained from scratch achieving **97% classification accuracy**
- 📊 Custom dataset built by collecting and labeling hand landmark data
- ⏱️ Time-based stabilization to reduce noisy predictions
- 📝 Real-time sentence generation from recognized letters
- 🤖 Local AI interaction using Ollama — fully offline, no external APIs
- 💾 Automatic sentence submission after hand absence detection

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | TensorFlow, Keras, Scikit-learn |
| AI Integration | Ollama, Llama 3.1 8B |

---

## Neural Network

The classification model is a **Feed-Forward Neural Network (FFNN)** that takes hand landmark coordinates extracted by MediaPipe as input and predicts the corresponding ASL letter (A–Z).

- Dataset created from scratch for this project
- Trained and evaluated using TensorFlow/Keras
- Achieved **~97% accuracy** on test data

---

## Installation

### Prerequisites

- Python 3.8+
- Webcam
- [Ollama](https://ollama.com) installed and running locally

### 1. Clone the repository

```bash
git clone https://github.com/Abdelali516/ASL-letters-A-Z.git
cd ASL-letters-A-Z
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe tensorflow scikit-learn numpy
```

### 3. Pull the Llama model via Ollama

```bash
ollama pull llama3.1:8b
```

### 4. Run the application

```bash
python main.py
```

---

## How It Works

```
Webcam Feed
    ↓
MediaPipe Hand Landmark Detection
    ↓
FFNN Gesture Classification (97% accuracy)
    ↓
Real-Time Sentence Building
    ↓
Hand Absence Detected → Sentence Submitted
    ↓
Llama 3.1 8B (via Ollama) → AI Response
```

---

## Special Cases

Some ASL letters involve **motion** rather than static hand shapes:

| Letter | Status |
|---|---|
| A–I, K–Y | ✅ Fully supported |
| J, Z | ⚠️ Represented as static placeholders (motion-based recognition planned) |

---

## Future Improvements

- [ ] Motion-based recognition for dynamic letters (J and Z)
- [ ] Temporal voting for improved prediction stability
- [ ] Graphical interface to display the recognized sentence
- [ ] Expanded dataset for higher accuracy
- [ ] Support for full ASL words and phrases beyond the alphabet

---

## Purpose

This project demonstrates how **computer vision**, **machine learning**, and **local AI models** can be combined into an interactive real-time system — making sign language more accessible while keeping all processing fully local and private.

---

## Author

**Abdelali Elasbi**
- GitHub: [@Abdelali516](https://github.com/Abdelali516)
- LinkedIn: [linkedin.com/in/abdelali-elasbi](https://linkedin.com/in/abdelali-elasbi)
