import os
import numpy as np
import tensorflow as tf
import json
from sklearn.utils import class_weight

# Define the path to your dataset and the directory to save/load preprocessed data
preprocessed_data_path = 'E:/Semester 5/Integrated Design Project/Test/Dataset/preprocessed_data'
model_save_path = 'E:/Semester 5/Integrated Design Project/Test/Dataset/sign_language_model.h5'

# Load the JSON files
with open('E:/Semester 5/Integrated Design Project/Test/Dataset/MSASL_classes.json', 'r') as f:
    classes = json.load(f)

# Load preprocessed data
print("Loading preprocessed data...")
X_train = np.load(os.path.join(preprocessed_data_path, 'X_train.npy'))
y_train = np.load(os.path.join(preprocessed_data_path, 'y_train.npy'))
X_validate = np.load(os.path.join(preprocessed_data_path, 'X_validate.npy'))
y_validate = np.load(os.path.join(preprocessed_data_path, 'y_validate.npy'))
X_test = np.load(os.path.join(preprocessed_data_path, 'X_test.npy'))
y_test = np.load(os.path.join(preprocessed_data_path, 'y_test.npy'))

# Check the shapes of the preprocessed data
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_validate.shape}, {y_validate.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# Calculate class weights
print("Calculating class weights...")
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights = {i: float(class_weights[i]) for i in range(len(class_weights))}
print(f"Class weights: {class_weights}")

# Build an improved model with more layers and regularization
print("Building the model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 63)),  # Sequence length is variable, 63 landmarks per frame
    tf.keras.layers.Conv1D(128, 3, activation='relu'),  # Convolutional layer for spatial features
    tf.keras.layers.BatchNormalization(),  # Batch normalization for faster convergence
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),  # Bidirectional LSTM
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization
    tf.keras.layers.Dense(len(classes), activation='softmax')  # Output layer
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adjusted learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with class weights
print("Starting training...")
if X_train.size > 0 and y_train.size > 0:
    history = model.fit(X_train, y_train, validation_data=(X_validate, y_validate), 
                        epochs=100, batch_size=32, class_weight=class_weights)
else:
    print("Training data is empty. Please check the dataset and paths.")

# Evaluate the model
print("Evaluating the model...")
if X_test.size > 0 and y_test.size > 0:
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')
else:
    print("Test data is empty. Please check the dataset and paths.")

# Save the model
print(f"Saving the model to {model_save_path}...")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
