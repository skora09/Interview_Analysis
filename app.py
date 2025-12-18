import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from collections import deque
import numpy as np
import pandas as pd
import seaborn as sns
import time
import threading
import mss


CAPTURE_CONFIG = {
    "top": 50,
    "left": 0,
    "width": 800,
    "height": 600
}


WINDOW_TITLE = "PREVIEW - Drag away from capture area"


TARGET_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
CHART_COLORS = {
    'angry': 'red',
    'disgust': 'green',
    'fear': 'purple',
    'happy': 'yellow',
    'sad': 'blue',
    'surprise': 'orange',
    'neutral': 'gray'
}
VIDEO_COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 128, 0),
    'fear': (128, 0, 128),
    'happy': (0, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (0, 165, 255),
    'neutral': (200, 200, 200)
}

ANALYSIS_INTERVAL = 0.5
HISTORY_SIZE = 50


recent_data = {
    emo: deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
    for emo in TARGET_EMOTIONS
}
session_history = []
current_result = []
ai_lock = False


plt.ion()
fig, ax = plt.subplots(figsize=(6, 4))
lines = {}
x_data = np.arange(HISTORY_SIZE)

for emo in TARGET_EMOTIONS:
    line, = ax.plot(
        x_data,
        recent_data[emo],
        label=emo.capitalize(),
        color=CHART_COLORS[emo],
        linewidth=1.5
    )
    lines[emo] = line

ax.set_ylim(0, 100)
ax.set_title("Real-Time Screen Analysis")
ax.legend(loc='upper left', fontsize='x-small')
plt.tight_layout()


def analyze_frame(frame_bgr):
    global current_result, ai_lock
    try:
        objs = DeepFace.analyze(
            frame_bgr,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        current_result = objs
    except:
        pass
    finally:
        ai_lock = False


print("\n>>> SYSTEM STARTED <<<")
print("1. Move the Teams/Zoom window to the TOP-LEFT corner.")
print("2. The Preview window will open on the right.")
print("3. Press 'q' in the Preview window to exit.")

sct = mss.mss()
last_analysis_time = time.time()


cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_TITLE, 900, 50)
cv2.resizeWindow(WINDOW_TITLE, 600, 450)

try:
    while True:

        screenshot = np.array(sct.grab(CAPTURE_CONFIG))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        now = time.time()
        if (now - last_analysis_time) > ANALYSIS_INTERVAL and not ai_lock:
            ai_lock = True
            last_analysis_time = now
            frame_copy = frame.copy()
            threading.Thread(
                target=analyze_frame,
                args=(frame_copy,)
            ).start()

        if current_result and isinstance(current_result, list):
            detected_faces = 0
            emotion_sum = {e: 0.0 for e in TARGET_EMOTIONS}

            for face_obj in current_result:
                if face_obj.get('region', {}).get('w', 0) > 0:
                    detected_faces += 1
                    r = face_obj['region']
                    cv2.rectangle(
                        frame,
                        (r['x'], r['y']),
                        (r['x'] + r['w'], r['y'] + r['h']),
                        (0, 255, 0),
                        2
                    )
                    for emo in TARGET_EMOTIONS:
                        emotion_sum[emo] += face_obj['emotion'].get(emo, 0)

            if detected_faces > 0:
                avg = {
                    e: emotion_sum[e] / detected_faces
                    for e in TARGET_EMOTIONS
                }

                timestamp = time.time()
                record = {'timestamp': timestamp}

                for emo in TARGET_EMOTIONS:
                    recent_data[emo].append(avg[emo])
                    lines[emo].set_ydata(recent_data[emo])
                    record[emo] = avg[emo]

                session_history.append(record)

                dominant = max(avg, key=avg.get)
                cv2.putText(
                    frame,
                    f"{dominant.upper()}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    VIDEO_COLORS.get(dominant),
                    2
                )
            else:
                for emo in TARGET_EMOTIONS:
                    recent_data[emo].append(0)
                    lines[emo].set_ydata(recent_data[emo])
        else:
            for emo in TARGET_EMOTIONS:
                recent_data[emo].append(0)
                lines[emo].set_ydata(recent_data[emo])

        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except:
            pass

        cv2.imshow(WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    cv2.destroyAllWindows()
    plt.close()

    if session_history:
        print("Saving charts...")
        df = pd.DataFrame(session_history)
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]

        plt.figure(figsize=(10, 5))
        for emo in TARGET_EMOTIONS:
            plt.plot(df['time'], df[emo], label=emo, color=CHART_COLORS[emo])
        plt.title("Screen History")
        plt.legend()
        plt.savefig('final_timeline.png')

        plt.figure(figsize=(6, 4))
        pos = df['happy'].sum()
        neg = df[['angry', 'fear', 'sad', 'disgust']].sum().sum()
        neu = df[['neutral', 'surprise']].sum().sum()
        sns.barplot(x=['Positive', 'Negative', 'Neutral'], y=[pos, neg, neu])
        plt.savefig('final_balance.png')

        print("Done.")
        df.to_csv('dados_reuniao.csv', index=False)
