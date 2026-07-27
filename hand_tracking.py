import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import socket

# Set up the hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

class Robotfinger:

    def __init__(self, landmark1, landmark2, landmark0, max_value, min_value):
        self.tip_x = landmark1.x
        self.tip_y = landmark1.y
        self.base_x = landmark2.x
        self.base_y = landmark2.y
        self.wrist_x = landmark0.x
        self.wrist_y = landmark0.y
        self.max = max_value
        self.min = min_value


    def normalize_finger_distance_bend(self):
        bend_distance = ((self.tip_x - self.base_x)**2 + (self.tip_y - self.base_y)**2) ** 0.5
        fixed_distance = ((self.base_x - self.wrist_x)**2 + (self.base_y - self.wrist_y)**2)**0.5
        return bend_distance/fixed_distance

    def transformation_bend(self, distance):                                       # function use for tranforming value of distance into numbers in range 0 - 1, 
        normalized = (distance - self.min) / (self.max - self.min)                 # BUT EXCEED VALUE (LARGER THAN 1 AND LESS THAN 0 STILL BE RECORDED AND PUBLISHED)
        return normalized

    def clamp_value(self, distance):
        if distance > 1:
            return 1
        elif distance < 0:
            return 0
        else:
            return distance

#PI_IP = "192.168.1.238"
#client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client.connect((PI_IP, 9999))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to MediaPipe's image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run hand detection
    result = detector.detect(mp_image)

    # Draw landmarks if a hand was detected
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
        
    
            thumb = Robotfinger(hand_landmarks[4], hand_landmarks[2], hand_landmarks[0], 0.600, 0.400)    
            bend1 = thumb.normalize_finger_distance_bend()
            true_value1 = thumb.transformation_bend(bend1)
            clamp1 = thumb.clamp_value(true_value1)


            index = Robotfinger(hand_landmarks[8], hand_landmarks[5], hand_landmarks[0], 0.836, 0.017)
            bend2 = index.normalize_finger_distance_bend()
            true_value2 = index.transformation_bend(bend2)
            clamp2 = index.clamp_value(true_value2)


            middle = Robotfinger(hand_landmarks[12], hand_landmarks[9], hand_landmarks[0], 0.938, 0.022)
            bend3 = middle.normalize_finger_distance_bend()
            true_value3 = middle.transformation_bend(bend3)
            clamp3 = middle.clamp_value(true_value3)


            ring = Robotfinger(hand_landmarks[16], hand_landmarks[13], hand_landmarks[0], 0.950, 0.021)
            bend4 = ring.normalize_finger_distance_bend()
            true_value4 = ring.transformation_bend(bend4)
            clamp4 = ring.clamp_value(true_value4)


            pinky = Robotfinger(hand_landmarks[20], hand_landmarks[17], hand_landmarks[0], 0.840, 0.021)
            bend5 = pinky.normalize_finger_distance_bend()
            true_value5 = pinky.transformation_bend(bend5)
            clamp5 = pinky.clamp_value(true_value5)
            

            cv2.putText(frame, f"Thumb: {round(clamp1, 3)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Index: {round(clamp2, 3)}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Middle: {round(clamp3, 3)}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Ring: {round(clamp4, 3)}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pinky: {round(clamp5, 3)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
            #message = f"{clamp1},{clamp2},{clamp3},{clamp4},{clamp5}"
            #client.sendall(message.encode())

            
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 250, 0), -1)

                
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 