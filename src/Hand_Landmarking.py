#imports 
import cv2
import mediapipe as mp #don't forget to install using pip

class landmarker_and_result():
    def __init__(self):
        self.result = mp.tasks.vision.HandLandmarker
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.createLandmarker
    
    def createLandmarker(self):
        # callback function
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
        options = mp.tasks.vision.HandLandmarkerOptions( 
            base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
            num_hands = 2, # track both hands
            min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
            min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
            min_tracking_confidence = 0.3, # lower than value to get predictions more often
            result_callback=update_result)
  
        # initialize landmarker
        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

    def close(self):
        # close landmarker
        self.landmarker.close()



def main():
#access cam    
    cap = cv2.VideoCapture(0)

    while True: 
        #get frame
        ret, frame = cap.read()
        
        #flip frame (mirroring)
        frame = cv2.flip(frame, 1)

        #display frame 
        cv2.imshow('frame', frame)
        if cv2.waitKey(1)==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()