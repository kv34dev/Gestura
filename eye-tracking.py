import cv2
import mediapipe as mp
import math

mp_face = mp.solutions.face_mesh

# Индексы точек вокруг глаз (Mediapipe FaceMesh)
LEFT_EYE = [33, 133]     # крайние точки левого глаза
RIGHT_EYE = [362, 263]   # крайние точки правого глаза

cap = cv2.VideoCapture(0)

with mp_face.FaceMesh(
        max_num_faces=1,
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
            face = results.multi_face_landmarks[0]

            # === ЛЕВЫЙ ГЛАЗ ===
            lx1 = int(face.landmark[LEFT_EYE[0]].x * w)
            ly1 = int(face.landmark[LEFT_EYE[0]].y * h)
            lx2 = int(face.landmark[LEFT_EYE[1]].x * w)
            ly2 = int(face.landmark[LEFT_EYE[1]].y * h)

            left_center = ((lx1 + lx2) // 2, (ly1 + ly2) // 2)
            left_radius = int(math.dist((lx1, ly1), (lx2, ly2)) * 1.2)

            cv2.circle(frame, left_center, left_radius, (0, 255, 0), 2)

            # === ПРАВЫЙ ГЛАЗ ===
            rx1 = int(face.landmark[RIGHT_EYE[0]].x * w)
            ry1 = int(face.landmark[RIGHT_EYE[0]].y * h)
            rx2 = int(face.landmark[RIGHT_EYE[1]].x * w)
            ry2 = int(face.landmark[RIGHT_EYE[1]].y * h)

            right_center = ((rx1 + rx2) // 2, (ry1 + ry2) // 2)
            right_radius = int(math.dist((rx1, ry1), (rx2, ry2)) * 1.2)

            cv2.circle(frame, right_center, right_radius, (0, 255, 0), 2)

        cv2.imshow("Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
