from ultralytics import YOLO
import cvzone
import cv2
import math
import pygame
from plyer import notification

# Initialize Pygame for sound
pygame.mixer.init()
alarm_sound_path = r'C:\Users\deepi\Desktop\FIRE\alarm_sound.mp3'  # Replace with the actual path
alarm_sound = pygame.mixer.Sound(alarm_sound_path)

# Running real-time from webcam
cap = cv2.VideoCapture(0)  # 0 is the default index for the webcam
model = YOLO('fire.pt')

# Reading the classes
classnames = ['fire']

# Flag to track if alarm is already playing
alarm_playing = False

# Flag to track if notification has been sent
notification_sent = False

while True:
    ret, frame = cap.read()

    result = model(frame, stream=True)

    # Getting bbox, confidence, and class names information to work with
    fire_detected = False
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 60 and classnames[Class] == 'fire':
                fire_detected = True
                if not notification_sent:
                    # Send notification when fire is detected
                    notification_sent = True
                    notification.notify(
                        title='Fire Detected',
                        message='There is a fire in the vicinity!',
                        app_name='Fire Detection System',
                        timeout=10  # Notification will disappear after 10 seconds
                    )

                # Draw red box with percentage
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    if fire_detected:
        # Play alarm sound if fire is detected
        if not alarm_playing:
            print("Playing alarm sound...")
            pygame.mixer.Channel(0).play(alarm_sound)
            alarm_playing = True
    else:
        # Reset notification_sent and alarm_playing flags when fire is not detected
        notification_sent = False
        if alarm_playing:
            print("Stopping alarm sound...")
            pygame.mixer.Channel(0).stop()
            alarm_playing = False

    cv2.imshow('frame', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
