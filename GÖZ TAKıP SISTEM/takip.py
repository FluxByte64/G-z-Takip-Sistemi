import cv2
import mediapipe as mp
from playsound import playsound
import threading
from collections import deque
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def play_alarm():
    playsound("ireland-eas-alarm-264351.mp3")

def is_eye_closed(landmarks, left_eye_indices, right_eye_indices):
    def eye_aspect_ratio(eye):
        vertical1 = ((eye[1][0] - eye[5][0])**2 + (eye[1][1] - eye[5][1])**2) ** 0.5
        vertical2 = ((eye[2][0] - eye[4][0])**2 + (eye[2][1] - eye[4][1])**2) ** 0.5
        horizontal = ((eye[0][0] - eye[3][0])**2 + (eye[0][1] - eye[3][1])**2) ** 0.5
        return (vertical1 + vertical2) / (2.0 * horizontal)
    
    left_eye = [(landmarks[idx].x, landmarks[idx].y) for idx in left_eye_indices]
    right_eye = [(landmarks[idx].x, landmarks[idx].y) for idx in right_eye_indices]

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    return left_ear, right_ear

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25  
CLOSED_FRAMES_THRESHOLD = 15  

ear_history = deque(maxlen=30)
alarm_triggered = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear, right_ear = is_eye_closed(face_landmarks.landmark, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)
            avg_ear = (left_ear + right_ear) / 2.0

            ear_history.append(avg_ear)

            closed_frames = sum(ear < EAR_THRESHOLD for ear in ear_history)

            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not alarm_triggered:
                cv2.putText(frame, "Goz Kapali! ALARM!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                alarm_triggered = True
                threading.Thread(target=play_alarm, daemon=True).start()

            elif closed_frames < CLOSED_FRAMES_THRESHOLD:
                alarm_triggered = False

    cv2.imshow('Göz Takip Sistemi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
