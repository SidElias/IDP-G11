import speech_recognition as sr
import pyttsx3 

# Initialize 
r = sr.Recognizer() 
def SpeakText(command):
    
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()
    
# Loop infinitely
while(1): 
    try:
        # use the microphone as source for input.
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            
            #listens for the user's input 
            audio2 = r.listen(source2)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            print("Did you say: ",MyText)
            SpeakText(MyText)
            
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        
    except sr.UnknownValueError:
        print("unknown error occurred")
