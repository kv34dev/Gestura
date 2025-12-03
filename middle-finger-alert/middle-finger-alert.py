import cv2
import mediapipe as mp
import simpleaudio as sa

# mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# video stream
cap = cv2.VideoCapture(0)

# load wav once
wave_obj = sa.WaveObject.from_wave_file("alert2.wav")
play_obj = None  # track current playback

# check if only middle finger up
def is_middle_finger_up(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    fingers_up = []
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

    return fingers_up == [0, 1, 0, 0]

with mp_hands.Hands(
    max_num_hands=10,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # flip horizontally (mirror)

        # convert to rgb
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        middle_finger_up = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if is_middle_finger_up(hand_landmarks):
                    middle_finger_up = True

        # play sound
        if middle_finger_up:
            if play_obj is None or not play_obj.is_playing():
                play_obj = wave_obj.play()
        else:
            if play_obj is not None and play_obj.is_playing():
                play_obj.stop()
                play_obj = None

        cv2.imshow('Middle Finger Detector', image)

        if cv2.waitKey(1) & 0xFF == 27:
            if play_obj is not None and play_obj.is_playing():
                play_obj.stop()
            break

cap.release()
cv2.destroyAllWindows()
