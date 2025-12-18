# Real-Time Executive Emotional Intelligence Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red) ![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green) ![Status](https://img.shields.io/badge/Status-Prototype-orange)

## 📋 Project Overview

This project is a Proof of Concept (PoC) designed to analyze the emotional tone of virtual interview in real-time. Unlike standard tools that rely on text transcription, this solution uses **Computer Vision** to capture non-verbal cues (micro-expressions) from the video feed of conference platforms (Zoom, Teams, Google Meet).

The system consists of two main components:
1.  **Capture & Inference Engine (`app.py`):** Non-intrusive screen capture that processes video feeds using Deep Learning to detect emotional states without accessing the meeting API directly.
2.  **Executive Dashboard (`analysis.py`):** A Streamlit-based post-meeting analytics tool that translates raw emotional data into business insights (Productivity, Focus, and Tension levels).

## 🛠️ Architecture & Tech Stack

* **Core Language:** Python
* **Computer Vision:** OpenCV (`cv2`), MSS (Fast Screen Capture)
* **Deep Learning:** DeepFace (wrapping pre-trained models for facial emotion recognition)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib (Real-time), Plotly & Streamlit (Dashboard)
* **Concurrency:** Threading (to decouple UI rendering from heavy model inference)

## 🚀 Key Features

* **Non-Intrusive Monitoring:** Uses screen scraping (`mss`) rather than webcam injection, allowing analysis of remote participants displayed on the screen.
* **Real-Time Feedback:** Displays a live graph overlay showing the immediate emotional trend of the meeting.
* **Business Logic Mapping:** Raw emotions are reinterpreted for a corporate context (e.g., "Disgust" micro-expressions often correlate with deep "Focus/Scrutiny"; "Sadness" often correlates with "Thoughtfulness").
* **Automated Reporting:** Generates a post-meeting summary with actionable verdicts (e.g., "Highly Productive" vs. "Conflict-Prone").

## 📂 Project Structure

bash
├── app.py                # Main script: Screen capture and Real-time Inference
├── analysis.py           # Dashboard: Streamlit application for post-analysis
├── requirements.txt      # Project dependencies
├── dados_reuniao.csv     # Generated data (output of app.py)
└── README.md             # Documentation



 How to Run
Prerequisites
Ensure you have Python installed.

Bash

pip install -r requirements.txt


1. Running the Capture Engine
Start the real-time analyzer during a meeting.

Bash

python app.py
Setup: A capture window (800x600) is defined at the top-left of your screen. Move your Teams/Zoom window into this area.
Preview: A separate preview window will show what the algorithm sees.
Stop: Press q in the preview window to stop recording and generate the data (dados_reuniao.csv).



2. Viewing the Analytics Dashboard
After the meeting, visualize the results.

Bash

streamlit run analysis.py


 Logical Approach
1. Data Acquisition
The system uses mss to grab screenshots at a high frame rate from a specific region of the monitor. This allows the tool to be platform-agnostic (works with any video conferencing software).

2. Emotion Inference
To maintain high performance, the heavy lifting (DeepFace analysis) runs on a separate Daemon Thread. This prevents the video feed from freezing while the neural network processes the frame.

Sampling Rate: Analysis is performed every 0.5 seconds to balance CPU load and temporal resolution.

Smoothing: A deque (double-ended queue) with a history of 50 frames is used to smooth out jittery predictions, providing a cleaner trend line.


3. Contextual Translation (The "Business Layer")
In analysis.py, we apply domain knowledge to raw outputs:

Anger + Fear → Tension/Stress Index

Disgust (often nose wrinkling) → Remapped to Focus/Concentration

Happy → Well-being/Engagement


This translation turns abstract AI outputs into understandable HR/Management metrics.

⚠️ Disclaimer
This project uses the DeepFace library which relies on pre-trained models (VGG-Face, Google FaceNet, etc.). It is intended for research and demonstration purposes only. Ethical considerations regarding consent and privacy should be strictly observed when recording or analyzing meetings.
