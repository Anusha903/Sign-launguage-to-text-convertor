import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Step 1: Load and Preprocess Dataset ---
DATASET_PATH = 'dataset'  # Path to ASL image folders
IMAGE_SIZE = (64, 64)

data = []
labels = []

print("[INFO] Loading images...")
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                image = load_img(img_path, target_size=IMAGE_SIZE, color_mode='grayscale')
                image = img_to_array(image)
                image = image / 255.0  # Normalize
                data.append(image)
                labels.append(folder)
            except:
                continue

# --- Step 2: Encode Labels and Split ---
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# --- Step 3: Build CNN Model ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(lb.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Step 4: Train ---
print("[INFO] Training model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# --- Step 5: Evaluate ---
loss, acc = model.evaluate(X_test, y_test)
print(f"[INFO] Test accuracy: {acc*100:.2f}%")

# --- Step 6: Save Model & Labels ---
model.save("asl_model.h5")
np.save("label_classes.npy", lb.classes_)
print("[INFO] Model and labels saved.")

# --- Step 7: Plot Training History ---
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
