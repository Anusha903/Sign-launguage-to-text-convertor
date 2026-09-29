import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and label classes
model = load_model("asl_model.h5")
label_classes = np.load("label_classes.npy")

IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence to accept a prediction

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("[INFO] Real-time ASL detection started. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess for prediction
    gray = cv2.cvtColor(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    reshaped = np.reshape(norm, (1, IMG_SIZE, IMG_SIZE, 1))

    # Predict
    prediction = model.predict(reshaped)
    label_index = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_char = label_classes[label_index] if confidence > CONFIDENCE_THRESHOLD else "Blank"

    # Display
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{predicted_char} ({confidence*100:.1f}%)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ASL Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
