import os
import cv2
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import threading
import time


# 1. Initialize the Smooth Voice Function
def speak(text):
    def talk():
        # Create the high-quality audio file
        tts = gTTS(text=text, lang='en', tld='com')
        tts.save("speech.mp3")
        # Play it using a system-level player (no choppiness!)
        os.system("mpg123 -q speech.mp3")

    threading.Thread(target=talk).start()


# 2. Setup the "Eye" (Same as before)
model = YOLO('yolov8n-seg.pt')
CALIBRATION_CONSTANT = 540


def run_smooth_jarvis():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    last_speech_time = 0
    speech_cooldown = 5

    print("🎙️ Smooth Jarvis Protocol: ONLINE")

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(source=frame, show=False, device='cpu',
                                verbose=False, conf=0.5)

        for result in results:
            annotated_frame = result.plot()

            if result.masks is not None:
                labels = [model.names[int(cls)] for cls in result.boxes.cls]

                for i, mask in enumerate(result.masks.xy):
                    label = result.names[int(result.boxes.cls[i])]
                    area = cv2.contourArea(mask.astype(np.float32))
                    dist = CALIBRATION_CONSTANT / np.sqrt(area)

                    current_time = time.time()
                    if current_time - last_speech_time > speech_cooldown:
                        # 🦜 Cosmo Alert
                        if label == 'bird':
                            speak("Air support detected. Cosmo is in position.")
                            last_speech_time = current_time
                        # 🐶 Thor Alert
                        elif label == 'dog' and dist < 2.5:
                            speak("Thor is approaching the workspace. Maintain distance.")
                            last_speech_time = current_time

            cv2.imshow("Jarvis Smooth Monitor", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_smooth_jarvis()