import cv2
import numpy as np
from pygame import mixer
import os

# Initialize pygame mixer
mixer.init()

# Ensure sound files exist
alarm_path = '/Users/nishashukla/Downloads/Drowsiness Detection/alert.wav'
alert_path = '/Users/nishashukla/Downloads/Drowsiness Detection/alert.wav'

if not os.path.exists(alarm_path) or not os.path.exists(alert_path):
    print("Error: Sound files not found!")
    exit()
    

sound1 = mixer.Sound(alarm_path)  # Sound for drowsiness
sound2 = mixer.Sound(alert_path)  # Sound for yawning

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier("/Users/nishashukla/Downloads/Drowsiness Detection/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/Users/nishashukla/Downloads/Drowsiness Detection/haarcascade_mcs_mouth.xml")

mouth_cascade_path = r"/Users/nishashukla/Downloads/Drowsiness Detection/haarcascade_mcs_mouth.xml"

# Load the cascade classifier
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Increase FPS for smoother video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Thresholds
EYE_AR_CONSEC_FRAMES = 10  # Reduced for faster alert
MOUTH_AR_CONSEC_FRAMES = 10  # Yawning threshold
COUNTER_EYES = 0
COUNTER_MOUTH = 0
alarm_status = False
yawn_status = False

def alarm():
    """Trigger alarm if eyes are closed for too long."""
    global alarm_status
    if not alarm_status:
        alarm_status = True
        sound1.play()

def yawn_alert():
    """Trigger alert if yawning is detected."""
    global yawn_status
    if not yawn_status:
        yawn_status = True
        sound2.play()

def reset_alarm():
    """Stop alarm when eyes are open."""
    global alarm_status
    alarm_status = False
    sound1.stop()

def reset_yawn_alert():
    """Stop yawn alert when mouth is closed."""
    global yawn_status
    yawn_status = False
    sound2.stop()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7, minSize=(20, 20))

        # Detect mouth
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=20, minSize=(30, 30))

        # Eye-based drowsiness detection
        if len(eyes) == 0:
            COUNTER_EYES += 1
            if COUNTER_EYES >= EYE_AR_CONSEC_FRAMES:
                alarm()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            COUNTER_EYES = 0
            reset_alarm()

        # Yawning detection (Mouth open for a long duration)
        if len(mouths) > 0:
            COUNTER_MOUTH += 1
            if COUNTER_MOUTH >= MOUTH_AR_CONSEC_FRAMES:
                yawn_alert()
                cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            COUNTER_MOUTH = 0
            reset_yawn_alert()

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
