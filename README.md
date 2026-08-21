This project tracks hand and finger movements in real time using a laptop webcam, OpenCV, and MediaPipe. Raw finger positions are naturally noisy, so I calculate each finger's bend using landmark distances between each fingertip and base, then normalize and clamp the result to a 0-1 range, and from those results I convert them into an appropriate value that can directly control a servo motor.



Planned Improvements:



DEBUGGING: Use Kalman filtering for smoother frame-to-frame tracking, thereby producing clean and stable values.

Run on a 3D hand model with servo control via Raspberry Pi GPIO

Send tracking data over a socket connection to a Raspberry Pi

