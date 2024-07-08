import json
import cv2
import os
from threading import Thread, Event
import pygame
import sys
import speech_recognition as sr
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = 'C:/Users/Asus/Documents/MJIIT/Sem5/IDP/Integrated Design Project/Test/Dataset/video/train'
ANNOTATIONS_DIR = 'C:/Users/Asus/Documents/MJIIT/Sem5/IDP/Integrated Design Project/Test/Dataset'
ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, 'MSASL_train.json')

# Ensure Pygame is initialized
pygame.init()

# Set to keep track of displayed words
displayed_words = set()

# Global flags to control termination and display stop
terminate_program = Event()
stop_display = Event()

# Function to extract frames from video and display them using Pygame
def extract_and_display_frames(video_path, start_time, end_time, fps=30):
    print(f"Extracting and displaying frames from {video_path} from {start_time} to {end_time}")
    cap = cv2.VideoCapture(video_path)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    # Get the original resolution of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a Pygame window with the original resolution
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Video Frame Display')

    while cap.isOpened() and not stop_display.is_set() and not terminate_program.is_set():
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break

        frame_count += 1

        # Convert the frame to a Pygame surface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

        # Display the frame in the Pygame window
        screen.blit(frame_surface, (0, 0))
        pygame.display.update()

        # Increase FPS by reducing delay
        pygame.time.delay(int(1000 / fps))

        # Check for events (including quit and key press)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                terminate_program.set()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Press 'q' to stop the display
                    cap.release()
                    pygame.quit()
                    stop_display.set()
                    return
                elif event.key == pygame.K_ESCAPE:  # Press 'Esc' to terminate the program
                    cap.release()
                    pygame.quit()
                    terminate_program.set()
                    sys.exit()  # Ensure immediate program termination

    cap.release()
    pygame.quit()

# Load JSON data
print(f"Loading annotations from {ANNOTATIONS_PATH}")
with open(ANNOTATIONS_PATH) as f:
    annotations = json.load(f)

# Function to recognize speech and display frames
def recognize_and_display():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    print("Ready to recognize speech.")

    while not terminate_program.is_set():
        with microphone as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5)

        try:
            print("Recognizing...")
            speech_text = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {speech_text}")

            # Check if the word has already been displayed
            if speech_text in displayed_words:
                print(f"Word '{speech_text}' has already been displayed.")
                continue

            # Add the word to the set of displayed words
            displayed_words.add(speech_text)

            # Find matching annotation
            matched_annotations = [ann for ann in annotations if ann['clean_text'] == speech_text]
            print(f"Found {len(matched_annotations)} matching annotations for '{speech_text}'")

            # Process each matched annotation
            for ann in matched_annotations:
                video_filename = ann['file'] + '.mp4'  # Assuming the file field matches the video filename
                video_path = os.path.join(VIDEOS_DIR, video_filename)
                print(f"Processing video: {video_path}")

                if not os.path.exists(video_path):
                    print(f"Video file {video_path} does not exist.")
                    continue

                start_time = ann['start_time']
                end_time = ann['end_time']

                # Reset stop_display flag before processing each annotation
                stop_display.clear()

                extract_and_display_frames(video_path, start_time, end_time)

                # Check if 'q' was pressed to stop displaying frames
                if stop_display.is_set():
                    break  # Exit processing this speech_text and go back to listening

        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except sr.WaitTimeoutError:
            print("Listening timed out, trying again...")

# Run speech recognition in a separate thread
speech_thread = Thread(target=recognize_and_display)
speech_thread.start()

# Main loop
try:
    while not terminate_program.is_set():
        pass
except KeyboardInterrupt:
    print("Program terminated by user.")

# Clean up Pygame and speech recognition on exit
pygame.quit()
terminate_program.set()  # Ensure speech recognition thread exits cleanly
speech_thread.join()
