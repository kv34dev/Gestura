import cv2
import mediapipe as mp

# камера
cap = cv2.VideoCapture(0)

# --- mediapipe modules ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,   # включает отслеживание радужки
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# drawing styles
white_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
green_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

cv2.namedWindow("Gestura", cv2.WINDOW_NORMAL)

# индексы радужки
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# соответствие кистей позе
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

# соответствие точек запястий из Hands
HAND_WRIST = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)
    pose_results = pose.process(rgb)

    h, w, _ = frame.shape

    # ----------- POSE (тело) ----------
    if pose_results.pose_landmarks:
        pose_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=5)
        mp_draw.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=pose_spec,
            connection_drawing_spec=pose_spec
        )

    # ----------- HANDS (руки) ----------
    hand_wrist_positions = []  # сохраняем координаты запястья для соединения с позой

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # координаты запястья
            wx = int(hand_landmarks.landmark[HAND_WRIST].x * w)
            wy = int(hand_landmarks.landmark[HAND_WRIST].y * h)
            hand_wrist_positions.append((wx, wy))

    # ----------- FACE + EYES (лицо + глаза зелёные точки) ----------
    if face_results.multi_face_landmarks:
        for face_lm in face_results.multi_face_landmarks:

            # лицо (tesselation + contours)
            mp_draw.draw_landmarks(
                frame, face_lm,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=white_spec,
                connection_drawing_spec=white_spec
            )
            mp_draw.draw_landmarks(
                frame, face_lm,
                mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=white_spec,
                connection_drawing_spec=white_spec
            )

            # зелёные точки на радужке
            for idx in LEFT_IRIS + RIGHT_IRIS:
                x = int(face_lm.landmark[idx].x * w)
                y = int(face_lm.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # отображение
    cv2.imshow("Gestura", frame)

    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("Gestura", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
