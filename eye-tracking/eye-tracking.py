import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

colors = [
    (0, 255, 0),     # green
    (0, 0, 255),     # red
    (255, 0, 0),     # blue
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (255, 128, 0),   # orange
    (128, 0, 255),   # purple
    (0, 128, 255),   # light blue
    (0, 255, 128)    # light green
]

with mp_face.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for i, face in enumerate(results.multi_face_landmarks):

                color = colors[i % len(colors)]  # каждому своё

                # pupil coords
                left_pupil  = face.landmark[468]
                right_pupil = face.landmark[473]

                lx = int(left_pupil.x * w)
                ly = int(left_pupil.y * h)
                rx = int(right_pupil.x * w)
                ry = int(right_pupil.y * h)

                # draw small circles
                cv2.circle(frame, (lx, ly), 5, color, -1)
                cv2.circle(frame, (rx, ry), 5, color, -1)

        cv2.imshow("Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
