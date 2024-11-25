import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the Sign Language MNIST dataset
train_df = pd.read_csv(r"D:\Monash Project\archive (2)\sign_mnist_train.csv")
test_df = pd.read_csv(r"D:\Monash Project\archive (2)\sign_mnist_test.csv")

# Separate features and labels
X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = train_df["label"].values

X_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_test = test_df["label"].values

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')  # 26 letters (A-Z)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    verbose=1
)

# Save the trained model
model.save("asl_model.h5")

