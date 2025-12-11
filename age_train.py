import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define age categories
age_categories = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60+"]

# Load and preprocess dataset
data_dir = "UTKFace"  # Change to your dataset path
images = []
ages = []

for file in os.listdir(data_dir):
    try:
        age = int(file.split("_")[0])  # Extract age from filename
        
        # Assign to age category
        if age <= 10:
            age_class = 0
        elif age <= 20:
            age_class = 1
        elif age <= 30:
            age_class = 2
        elif age <= 40:
            age_class = 3
        elif age <= 50:
            age_class = 4
        elif age <= 60:
            age_class = 5
        else:
            age_class = 6

        # Read and preprocess image
        img_path = os.path.join(data_dir, file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        image = cv2.resize(image, (48, 48))  # Resize to (48,48)
        image = image / 255.0  # Normalize

        images.append(image)
        ages.append(age_class)
    
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Convert lists to numpy arrays
images = np.array(images).reshape(-1, 48, 48, 1)
ages = to_categorical(ages, num_classes=7)  # One-hot encoding for age groups

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

# Define CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(7, activation="softmax")  # 7 output neurons for age groups
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    return model

# Train the model
model = build_model()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=6, batch_size=32)

# Save the model
model.save("age_prediction_model.h5")

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Accuracy Over Epochs")
plt.show()
