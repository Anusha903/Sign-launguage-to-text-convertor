import cv2
import os

# Config
DATASET_DIR = "dataset"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGES_PER_CLASS = 500
IMG_SIZE = 64

SPECIAL_CLASS = "blank"  # Special class for blank images

# Create dataset directory
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

current_label = None
count = 0
capturing = False
paused = False  # <-- Added pause flag

print("[INFO] Press a key (A-Z) to start capturing images for that letter.")
print("[INFO] Press '.' to capture blank images.")
print("[INFO] Press '1' to stop capturing.")
print("[INFO] Press '2' to pause/resume image capture.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    if capturing and current_label and not paused:  # <-- Skip saving if paused
        class_dir = os.path.join(DATASET_DIR, current_label)
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{current_label}_{count+1}.jpg")
        cv2.imwrite(img_path, gray)
        count += 1

        if count >= IMAGES_PER_CLASS:
            print(f"[INFO] Finished capturing for '{current_label}'")
            capturing = False
            current_label = None
            count = 0

    # Display info and box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if paused:
        status = "Paused..."
    else:
        status = f"Capturing '{current_label}': {count}/{IMAGES_PER_CLASS}" if capturing else "Waiting for key..."
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if paused else (255, 0, 0), 2)
    cv2.imshow("Sign Language Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):  # Stop capturing
        print(f"[INFO] Stopped capturing '{current_label}'")
        capturing = False
        current_label = None
        count = 0
    elif key == ord('2'):  # <-- Toggle pause
        paused = not paused
        print("[INFO] Paused" if paused else "[INFO] Resumed")
    elif chr(key).upper() in CLASSES and not paused:
        current_label = chr(key).upper()
        print(f"[INFO] Starting capture for '{current_label}'")
        count = 0
        capturing = True
    elif key == ord('.') and not paused:  # blank class
        current_label = SPECIAL_CLASS
        print(f"[INFO] Starting capture for '{current_label}'")
        count = 0
        capturing = True

    # Exit when webcam window is closed
    if cv2.getWindowProperty("Sign Language Capture", cv2.WND_PROP_VISIBLE) < 1:
        print("[INFO] Webcam window closed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
