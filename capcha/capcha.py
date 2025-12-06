import cv2
import mediapipe as mp
import random
import simpleaudio as sa

# sounds
ok_sound = sa.WaveObject.from_wave_file("ok.wav")
wrong_sound = sa.WaveObject.from_wave_file("wrong.wav")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# count fingers function (1–5)
def count_fingers(hand):
    tips = [4, 8, 12, 16, 20]
    pip = [3, 6, 10, 14, 18]

    count = 0
    for t, p in zip(tips[1:], pip[1:]):
        if hand.landmark[t].y < hand.landmark[p].y:
            count += 1
    if hand.landmark[tips[0]].x < hand.landmark[pip[0]].x:
        count += 1
    return count

# generate new 3-step captcha
def new_captcha():
    return [random.randint(1, 5) for _ in range(3)]

captcha = new_captcha()
step = 0
result_text = ""
cooldown = 0

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        finger_num = 0
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            finger_num = count_fingers(hand)

        if cooldown == 0:
            if step < 3:
                target = captcha[step]
                cv2.putText(frame, f"show number: {target}",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

                if finger_num == target and finger_num != 0:
                    ok_sound.play()
                    step += 1
                    cooldown = 30
                elif finger_num != 0 and finger_num != target:
                    wrong_sound.play()
                    result_text = "start again"
                    step = 0
                    captcha = new_captcha()
                    cooldown = 60
            else:
                result_text = "you are not a robot"
                cv2.putText(frame, result_text, (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cooldown -= 1

        cv2.putText(frame, result_text, (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow("gesture captcha", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
