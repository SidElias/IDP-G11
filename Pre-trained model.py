import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def find_video_file(video_dir, video_file):
    """ Recursively search for the video file in the directory """
    for root, _, files in os.walk(video_dir):
        if video_file in files:
            return os.path.join(root, video_file)
    return None

def extract_video_info(json_data, video_dir):
    video_paths = []
    labels = []
    for item in json_data:
        video_file = item['file'] + '.mp4' # Removed the '.mp4' extension
        video_path = find_video_file(video_dir, video_file)
        if video_path:
            video_paths.append(video_path)
            labels.append(item['label'])
    return video_paths, labels

def extract_frames(video_path, frame_rate=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps // frame_rate)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % interval == 0:
            frame = cv2.resize(frame, (64, 64))  # Resize frame to 64x64
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

def preprocess_videos(video_paths):
    X = []
    for video_path in video_paths:
        frames = extract_frames(video_path)
        if frames.size > 0:
            X.append(frames)
    return np.array(X)

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Directory containing dataset
dataset_dir = r'E:\Semester 5\Integrated Design Project\Test\Dataset'  # Replace with your dataset directory

# Load JSON files
train_json_path = os.path.join(dataset_dir, 'MSASL_train.json')
validate_json_path = os.path.join(dataset_dir, 'MSASL_val.json')
test_json_path = os.path.join(dataset_dir, 'MSASL_test.json')
classes_json_path = os.path.join(dataset_dir, 'MSASL_classes.json')

train_data = load_json(train_json_path)
validate_data = load_json(validate_json_path)
test_data = load_json(test_json_path)
classes_data = load_json(classes_json_path)

# Directory containing videos
video_base_dir = r'E:\Semester 5\Integrated Design Project\Test\Dataset\video'  # Replace with your new video directory

# Extract video paths and labels
train_video_dir = os.path.join(video_base_dir, 'train')
validate_video_dir = os.path.join(video_base_dir, 'validate')
test_video_dir = os.path.join(video_base_dir, 'test')

train_videos, train_labels = extract_video_info(train_data, train_video_dir)
validate_videos, validate_labels = extract_video_info(validate_data, validate_video_dir)
test_videos, test_labels = extract_video_info(test_data, test_video_dir)

# Check if any videos were not found
if not train_videos:
    print("No train videos found.")
if not validate_videos:
    print("No validation videos found.")
if not test_videos:
    print("No test videos found.")

# Preprocess videos
X_train = preprocess_videos(train_videos)
X_validate = preprocess_videos(validate_videos)
X_test = preprocess_videos(test_videos)

# Check if videos were preprocessed successfully
if X_train.size == 0:
    print("No train videos were preprocessed successfully.")
if X_validate.size == 0:
    print("No validation videos were preprocessed successfully.")
if X_test.size == 0:
    print("No test videos were preprocessed successfully.")

# Number of classes (based on your classes.json)
num_classes = len(classes_data)

# Convert labels to categorical
y_train = to_categorical(train_labels, num_classes=num_classes)
y_validate = to_categorical(validate_labels, num_classes=num_classes)
y_test = to_categorical(test_labels, num_classes=num_classes)

# Reshape X to the format required by Conv3D
if X_train.size > 0:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 64, 64, 3)
if X_validate.size > 0:
    X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], 64, 64, 3)
if X_test.size > 0:
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 64, 64, 3)

#Build model
input_shape = (None, 64, 64, 3)  # Adjust based on your frame extraction
model = build_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
if X_train.size > 0 and X_validate.size > 0:
    history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_validate, y_validate))

    # Plot accuracy graph
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Graph')
    plt.legend()
    plt.show()
else:
    print("Training data or validation data is not available.")

# Evaluate the model
if X_test.size > 0:
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')
else:
    print("Test data is not available.")

# Save the trained model
model.save('sign_language_model.keras')
print("Model saved successfully.")