import cv2
import math

# загружаем классификатор хаара
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# включаем веб-камеру
cap = cv2.VideoCapture(0)

# список цветов, можно добавить больше
colors = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 128, 255),
    (0, 128, 255),
    (255, 128, 0)
]

# список обнаруженных лиц: центр -> цвет
tracked_faces = []   # [(x_center, y_center, color, id)]

next_id = 1  # номер для face 1, face 2, ...

# функция для нахождения расстояния между точками
def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # находим лица
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    detected_centers = []  # новые центры кадров

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        detected_centers.append((cx, cy, x, y, w, h))

    # связываем новые лица с существующими по расстоянию
    for cx, cy, x, y, w, h in detected_centers:

        matched = False

        for face in tracked_faces:
            fx, fy, color, fid = face

            # если лицо примерно на том же месте → считаем тем же лицом
            if dist((cx, cy), (fx, fy)) < 80:
                # обновляем координаты
                face_index = tracked_faces.index(face)
                tracked_faces[face_index] = (cx, cy, color, fid)

                # рисуем квадрат
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"face {fid}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                matched = True
                break

        # если лицо новое — добавляем
        if not matched:
            color = colors[(next_id - 1) % len(colors)]
            tracked_faces.append((cx, cy, color, next_id))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"face {next_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            next_id += 1  # просто увеличиваем счетчик

    # показываем изображение
    cv2.imshow("face detector", frame)

    # выход по esc
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
