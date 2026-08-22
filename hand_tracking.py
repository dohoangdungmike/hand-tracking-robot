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


    def normalize_finger_distance_bend(self):                                                        # Function used for calculating RAW BEND value between each finger tip and base
                                                                                                     # Additionally, by comparing the finger bend value with the fixed bend value of the palm, the output will be more accurate 
        bend_distance = ((self.tip_x - self.base_x)**2 + (self.tip_y - self.base_y)**2) ** 0.5
        fixed_distance = ((self.base_x - self.wrist_x)**2 + (self.base_y - self.wrist_y)**2)**0.5
        return bend_distance/fixed_distance

    def transformation_bend(self, distance):                                       # Function used for transforming value of distance into numbers in range 0 - 1, 
        normalized = (distance - self.min) / (self.max - self.min)                 # BUT EXCEED VALUE (LARGER THAN 1 AND LESS THAN 0 STILL BE RECORDED AND PUBLISHED)
        return normalized

    def clamp_value(self, distance):
        if distance > 1:
            return 1
        elif distance < 0:
            return 0
        else:
            return distance
    

class Kalman_Filter:

    def __init__ (self, former_state, former_uncertainty, process_noise, measurement_noise):    
        self.former_state = former_state
        self.former_uncertainty = former_uncertainty
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        

        
    def predicted_next_state(self):
        return self.former_state                         # These are all the Kalman Filter functions THAT WERE NOT USED DIRECTLY, BUT THEY WILL BE THE MAIN CONTRIBUTIONS TO THE UPDATE FUNC

    def predicted_next_uncertainty(self):
        return self.former_uncertainty + self.process_noise

    def Kalman_gain(self, calculate_predicted_next_uncertainty):
        return calculate_predicted_next_uncertainty / (calculate_predicted_next_uncertainty + self.measurement_noise)

    def update_the_next_state(self, calculate_predicted_next_state, calculate_Kalman_Gain, OpenCV_measurement):
        return calculate_predicted_next_state + calculate_Kalman_Gain*(OpenCV_measurement - calculate_predicted_next_state)

    def update_the_next_uncertainty(self, calculate_Kalman_Gain, calculate_predicted_next_uncertainty):
        return (1 - calculate_Kalman_Gain) * calculate_predicted_next_uncertainty



    def update(self, measurement):                      # measurement will be contain the clamp value
        predicted_state = self.predicted_next_state()
        predicted_uncertainty = self.predicted_next_uncertainty()                 # Those white words act as a parameters
        kalman_gain = self.Kalman_gain(predicted_uncertainty)
        new_state = self.update_the_next_state(predicted_state, kalman_gain, measurement)               # measurement must be a clamp value
        new_uncertainty = self.update_the_next_uncertainty(kalman_gain, predicted_uncertainty)

        self.former_state = new_state              # SAVE for next frame 
        self.former_uncertainty = new_uncertainty   # SAVE for next frame

        return new_state

#PI_IP = "192.168.1.238"
#client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client.connect((PI_IP, 9999))

thumb_setup = Kalman_Filter(0.5, 1, 0.0001, 0.00075)
index_setup = Kalman_Filter(0.5, 1, 0.0001, 0.00075)         # The more you increase the measurement noise, the smoother the output value it gonna be
                                                             # More accurate estimation of the Index_bend after INCREASED THE MEASUREMENT NOISE
middle_setup = Kalman_Filter(0.5, 1, 0.0001, 0.00075)
ring_setup = Kalman_Filter(0.5, 1, 0.0001, 0.00075)
pinky_setup = Kalman_Filter(0.5, 1, 0.0001, 0.00075)

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
        
    
            thumb = Robotfinger(hand_landmarks[4], hand_landmarks[2], hand_landmarks[0], 0.81, 0.21)    
            bend1 = thumb.normalize_finger_distance_bend()
            true_value1 = thumb.transformation_bend(bend1)
            clamp1 = thumb.clamp_value(true_value1)


            # Kalman Filter

            update_index1 = thumb_setup.update(clamp1)
            print(f"Raw Thumb: {clamp1:.4f}  Filtered: {update_index1:.4f}")

            index = Robotfinger(hand_landmarks[8], hand_landmarks[5], hand_landmarks[0], 0.92, 0.02)
            bend2 = index.normalize_finger_distance_bend()
            true_value2 = index.transformation_bend(bend2)
            clamp2 = index.clamp_value(true_value2)
    

            # Kalman Filter
            
            update_index2 = index_setup.update(clamp2)
            print(f"Raw Index: {clamp2:.4f}  Filtered: {update_index2:.4f}")            # This line of code can be used to compare the output value before and after using Kalman Filter


            middle = Robotfinger(hand_landmarks[12], hand_landmarks[9], hand_landmarks[0], 1.15, 0.002)
            bend3 = middle.normalize_finger_distance_bend()
            true_value3 = middle.transformation_bend(bend3)
            clamp3 = middle.clamp_value(true_value3)


            # Kalman Filter

            update_index3 = middle_setup.update(clamp3)
            print(f"Raw Middle: {clamp3:.4f}  Filtered: {update_index3:.4f}")
            

            ring = Robotfinger(hand_landmarks[16], hand_landmarks[13], hand_landmarks[0], 1.06, 0.001)
            bend4 = ring.normalize_finger_distance_bend()
            true_value4 = ring.transformation_bend(bend4)
            clamp4 = ring.clamp_value(true_value4)


            # Kalman Filter

            update_index4 = ring_setup.update(clamp4)
            print(f"Raw Ring: {clamp4:.4f}  Filtered: {update_index4:.4f}")
            

            pinky = Robotfinger(hand_landmarks[20], hand_landmarks[17], hand_landmarks[0], 0.95, 0.017)
            bend5 = pinky.normalize_finger_distance_bend()
            true_value5 = pinky.transformation_bend(bend5)
            clamp5 = pinky.clamp_value(true_value5)
            

            # Kalman Filter

            update_index5 = pinky_setup.update(clamp5)
            print(f"Raw Pinky: {clamp5:.4f}  Filtered: {update_index5:.4f}")


            cv2.putText(frame, f"Thumb: {round(update_index1, 3)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Index: {round(update_index2, 3)}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Middle: {round(update_index3, 3)}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Ring: {round(update_index4, 3)}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pinky: {round(update_index5, 3)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
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
