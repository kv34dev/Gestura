import cv2
import mediapipe as mp

# камера
cap = cv2.VideoCapture(0)

# инициализация модулей mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# белый стиль линий для лица
white_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

while True:
    success, frame = cap.read()
    if not success:
        break

    # зеркальное отображение
    frame = cv2.flip(frame, 1)

    # перевод в rgb
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # обработка рук
    hand_results = hands.process(rgb)

    # обработка лица
    face_results = face_mesh.process(rgb)

    # ----- отрисовка рук -----
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

    # ----- отрисовка сетки лица -----
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:

            # сетка лица белыми линиями
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=white_spec,
                connection_drawing_spec=white_spec
            )

            # контуры лица белыми линиями
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=white_spec,
                connection_drawing_spec=white_spec
            )

    # показ изображения
    cv2.imshow("Hand + Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
