import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
# Since main.py is in the root directory, BASE_DIR is just this directory

MODEL_DIR   = 'model'
MODEL_PATH  = os.path.join(MODEL_DIR, 'sign_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.txt')
LANDMARKER_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')

# Load Model, Scaler and Labels
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}. Run training first!")
    sys.exit()

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(LABELS_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize MediaPipe (Tasks API)
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
#  FUNCTIONS
# ─────────────────────────────────────────────
def get_landmarks(img_rgb):
    """Extract and normalize landmarks from RGB image. Scale-Invariant."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        # 1. Translation Invariance
        wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
        
        # 2. Scale Invariance (Wrist to Middle Knuckle distance)
        m_knuckle = lm[9]
        dist = np.sqrt((m_knuckle.x - wrist_x)**2 + (m_knuckle.y - wrist_y)**2)
        if dist < 0.01: dist = 1.0
        
        coords = []
        for point in lm:
            coords.extend([
                (point.x - wrist_x) / dist, 
                (point.y - wrist_y) / dist, 
                (point.z - wrist_z) / dist
            ])
        return np.array(coords).reshape(1, -1), result.hand_landmarks[0]
    return None, None

def main():
    cap = cv2.VideoCapture(0)
    current_sentence = ""
    
    print("\n" + "="*50)
    print("  AI DEAF ASSISTANT — NATIVE OPENCV CAMERA APP")
    print("="*50)
    print("  Controls on Keyboard window:")
    print("    [S]       → Save highlighted letter")
    print("    [SPACE]   → Add space between words")
    print("    [Backspace] → Delete last letter")
    print("    [C]       → Clear whole sentence")
    print("    [Q] / Esc → Quit Application")
    print("="*50 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Mirror image for easier user Interaction
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hand
        features, landmarks = get_landmarks(rgb)
        
        prediction = "?"
        confidence = 0
        
        if features is not None:
            # Draw green landmark dots on fingers!
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            
            # Predict the sign using the trained AI
            X = scaler.transform(features)
            preds_proba = model.predict_proba(X)[0]
            pred_idx = np.argmax(preds_proba)
            confidence = preds_proba[pred_idx]
            
            if confidence > 0.6:
                prediction = class_names[pred_idx]
        
        # ── BEAUTIFUL UI OVERLAY ──
        # Top Left Box (Current Sign)
        box_color = (0, 200, 0) if confidence > 0.6 else (0, 0, 200)
        cv2.rectangle(frame, (10, 10), (280, 100), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 10), (280, 100), box_color, 2)
        cv2.putText(frame, f"SIGN: {prediction}", (20, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"CONF: {confidence*100:.1f}%", (20, 85), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)
        
        # Bottom Box (The Full Sentence)
        cv2.rectangle(frame, (0, h-70), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, f"WORD > {current_sentence}", (20, h-25), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

        # Show on screen
        cv2.imshow('AI Deaf Assistant (OpenCV View)', frame)
        
        # ── KEYBOARD CONTROLS ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # Esc
            break
        elif key == ord('s'):
            if prediction != "?" and prediction != 'nothing':
                if prediction == 'space':
                    current_sentence += " "
                elif prediction == 'del':
                    current_sentence = current_sentence[:-1]
                else:    
                    current_sentence += prediction
                print(f"[AI] Typed: {current_sentence}")
        elif key == ord(' '):
            current_sentence += " "
        elif key == 8: # Backspace
            current_sentence = current_sentence[:-1]
        elif key == ord('c'):
            current_sentence = ""

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
