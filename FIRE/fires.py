
from ultralytics import YOLO
import cvzone
import cv2
import math
import pygame

# Initialize Pygame for sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(r'C:\Users\deepi\Desktop\FIRE\alarm_sound.mp3')  # Replace with the actual path

# Running real-time from webcam
cap = cv2.VideoCapture(0)  # 0 is the default index for the webcam
model = YOLO('fire.pt')

# Reading the classes
classnames = ['fire']

# Flag to track if alarm is already playing
alarm_playing = False

while True:
    ret, frame = cap.read()

    # Optional: Resize the frame if needed
    # frame = cv2.resize(frame, (640, 480))

    result = model(frame, stream=True)

    # Getting bbox, confidence, and class names information to work with
    fire_detected = False
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 30 and classnames[Class] == 'fire':
                fire_detected = True
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    # Play alarm sound if fire is detected
    if fire_detected and not alarm_playing:
        pygame.mixer.Channel(0).play(alarm_sound)
        alarm_playing = True
    elif not fire_detected and alarm_playing:
        pygame.mixer.Channel(0).stop()
        alarm_playing = False

    cv2.imshow('frame', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
                                   #pygame.mixer.music.load(r'C:\Users\deepi\Desktop\FIRE\alarm_sound.mp3')
    