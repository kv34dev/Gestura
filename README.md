# Gestura

Gestura is a Python-based library designed for user interaction detection, focusing on gesture recognition and eye-tracking for anti-cheating purposes. It includes modules for verifying user actions and detecting middle finger gestures. 

## Features

- **CAPTCHA module**: CAPTCHA with sound feedback for verification.
- **Eye-tracking module**: Includes anti-cheating functionality for detecting suspicious user behavior through eye movements.
- **Middle-finger detection module**: Alerts when a middle finger gesture is detected.
- **Customizable alerts**: Sound alerts in WAV format for different types of events.
- Modular architecture — each component can run independently

## Running the System
Run full detection (face, body, hands)
```
python main.py
```
### Run individual modules:

Captcha:
```
python captcha/captcha.py
```
Eye-tracking:
```
python eye-tracking/eye-tracking.py
```
Anti-cheating eye monitoring:
```
python eye-tracking/anti-cheating.py
```
Middle-finger gesture detection:
```
python middle-finger-alert/middle-finger-alert.py
```


## Folder Structure
```
Gestura/
├── captcha/
│   ├── captcha.py               # Main CAPTCHA implementation
│   ├── ok.wav                   # Success alert sound
│   └── wrong.wav                # Error alert sound
├── eye-tracking/
│   ├── anti-cheating.py         # Anti-cheating functionality
│   └── eye-tracking.py          # Eye tracking functionality
├── middle-finger-alert/
│   ├── alert.wav                # Middle finger detection sound alert 1
│   ├── alert2.wav               # Middle finger detection sound alert 2
│   └── middle-finger-alert.py   # Middle finger detection logic
├── faces.py                     # Face recognition and detection module
├── main.py                      # Full pipeline: face, body and hand recognition
├── README.md                    # This file
```
