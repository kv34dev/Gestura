import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# iris indexes
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


def get_iris_center(landmarks, indexes, w, h):
    xs = [landmarks[i].x * w for i in indexes]
    ys = [landmarks[i].y * h for i in indexes]
    return sum(xs) / len(xs), sum(ys) / len(ys)


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

        # get iris centers
        lx, ly = get_iris_center(face.landmark, LEFT_IRIS, w, h)
        rx, ry = get_iris_center(face.landmark, RIGHT_IRIS, w, h)

        # draw for debugging
        cv2.circle(frame, (int(lx), int(ly)), 3, (0, 255, 0), -1)
        cv2.circle(frame, (int(rx), int(ry)), 3, (0, 255, 0), -1)

        # get face bounding box
        x_coords = [lm.x * w for lm in face.landmark]
        y_coords = [lm.y * h for lm in face.landmark]
        face_left, face_right = min(x_coords), max(x_coords)
        face_top, face_bottom = min(y_coords), max(y_coords)

        # thresholds 70% от ширины/высоты лица
        threshold_x = (face_right - face_left) * 0.4
        threshold_y = (face_bottom - face_top) * 0.4
        face_center_x = (face_right + face_left) / 2
        face_center_y = (face_bottom + face_top) / 2


        # проверка взгляда
        def is_gaze_away(ix, iy):
            return abs(ix - face_center_x) > threshold_x or abs(iy - face_center_y) > threshold_y


        if is_gaze_away(lx, ly) or is_gaze_away(rx, ry):
            print("gaze away from screen")

    cv2.imshow("Gaze Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
