# AI Deaf Assistant — Complete Setup Guide

Welcome to the AI Deaf Assistant project! This is a complete two-way communication system built with Flask, MediaPipe, and vanilla Javascript. It features real-time Sign Language Recognition (Camera), Text-to-Sign visualization, and Speech-to-Text capabilities.

## 🚀 Setup & Installation

Follow these steps to run the application perfectly on your local machine:

### 1. Install Dependencies
Open your terminal inside the project folder (`fyp-text-to-sign-language`) and run:
```bash
pip install -r requirements.txt
```

### 2. Fix Protobuf Conflict (IMPORTANT for Windows)
Google's MediaPipe requires a very specific version of the `protobuf` library to work smoothly on Windows without crashing. Run this command:
```bash
pip install protobuf==3.20.3
```

### 3. (Optional) Train the AI Model
*If you already have `sign_model.pkl` and `scaler.pkl` in the `model/` directory, you can skip this step!*
To train the model yourself:
```bash
python backend/train_model.py
```
*(Note: You can also easily run the `train_model.py` script on Google Colab or Kaggle. It is cloud-ready! Just download the resulting `.pkl` files and place them in your local `model/` folder.)*

### 4. Start the Application
Run the Flask server:
```bash
python backend/app.py
```

### 5. Open the Browser
Open **Google Chrome** (recommended for microphone API compatibility) and go exactly to:
👉 **http://localhost:5000** 👈
*(Note: Do not use 127.0.0.1 as it may block microphone permissions).*

---

## 🎮 How to Use the Features

### 🤟 1. Sign-to-Text (Camera AI)
1. In the browser, click **"Enable Camera"** on the left panel.
2. Form a sign in front of the camera.
3. Once the AI is confident (the box turns green), press **'S'** on your keyboard to save the letter.
4. Press the **Spacebar** to add spaces, or **'C'** to clear.
5. Click **"Paste to Chat"** to send your formed sentence!

### 👀 2. Text-to-Sign (Visualizer)
1. Type a message in the chat box (like "HELLO") and send it.
2. Hover your mouse over the sent message bubble.
3. Click the **"🤟 Visual Sign"** button that appears.
4. The visualizer at the top of the chat will smoothly animate through the corresponding hand sign images.

### 🎙️ 3. Speech-to-Text
1. Click the **"🎙️ Speak"** button near the text box.
2. Allow Chrome microphone permissions if asked.
3. Speak aloud, and the text will be automatically typed for you!
