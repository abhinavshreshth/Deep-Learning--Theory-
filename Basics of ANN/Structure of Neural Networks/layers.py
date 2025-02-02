import tensorflow as tf
from keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load and preprocess the dataset
# Download the dataset from: https://www.kaggle.com/uciml/pima-indians-diabetes-database
# Or use the following code to load it directly
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
data = pd.read_csv(url, names=column_names)

# Split into features (X) and target (y)
X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features (mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Build the neural network
model = tf.keras.Sequential(
    [
        layers.Input(shape=(8,)),  # Input layer with 8 features
        layers.Dense(16, activation="relu"),  # Hidden layer with 16 neurons
        layers.Dense(8, activation="relu"),  # Hidden layer with 8 neurons
        layers.Dense(
            1, activation="sigmoid"
        ),  # Output layer with 1 neuron (binary classification)
    ]
)

# Step 3: Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Step 4: Train the model
history = model.fit(
    X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=1
)

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Step 6: Make predictions
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(
    int
)  # Convert probabilities to binary predictions
print("Predictions:\n", predictions.flatten())
print("Actual Labels:\n", y_test)
