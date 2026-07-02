import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# ===============================
# GLOBAL EMOTION (FIX)
# ===============================
latest_emotion = "neutral"

# ===============================
# SESSION STATE INIT
# ===============================
if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0
    st.session_state.logs = []
    st.session_state.finished = False
    st.session_state.q_start_time = None

# ===============================
# Emotion CNN
# ===============================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

emotion_model = EmotionCNN()
emotion_model.load_state_dict(torch.load("models/emotion_cnn.pth", map_location="cpu"))
emotion_model.eval()

classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

# ===============================
# UI
# ===============================
st.title("🧠 Emotion-Guided Cognitive Risk Screening")

# ===============================
# STEP 1: LIVE EMOTION
# ===============================
st.subheader("📹 Step 1: Live Emotion Detection")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def video_frame_callback(frame):
    global latest_emotion

    img = frame.to_ndarray(format="bgr24")
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_full, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray_full[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48)) / 255.0
    else:
        face = cv2.resize(gray_full, (48,48)) / 255.0

    gray = torch.tensor(face, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(emotion_model(gray), dim=1)
        idx = probs.argmax().item()
        conf = probs.max().item()

    label = classes[idx]
    latest_emotion = label

    cv2.putText(img, f"{label} ({conf:.0%})", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback
)

if webrtc_ctx.state.playing:
    st.success("Camera active")
else:
    st.info("Turn ON camera")

# ===============================
# STEP 2: QUESTIONS
# ===============================
st.subheader("📝 Step 2: Behavioral Questions")

if not webrtc_ctx.state.playing:
    st.stop()

tasks = [
    {"q": "What would you most likely choose to do right now?",
     "options": ["Engage in a challenging task", "Watch something relaxing", "Talk to someone", "Take a break"]},

    {"q": "Which activity feels most appealing at this moment?",
     "options": ["Analyzing something", "Enjoying entertainment", "Scrolling casually", "Doing nothing"]},

    {"q": "How certain do you feel about your responses?",
     "options": ["Completely certain", "Partially certain", "Uncertain"]},

    {"q": "How much trust do you have in your decisions right now?",
     "options": ["Strong trust", "Moderate trust", "Low trust"]},

    {"q": "How mentally demanding do you find this activity?",
     "options": ["Not demanding", "Moderately demanding", "Highly demanding"]},

    {"q": "How easy is it for you to process information right now?",
     "options": ["Very easy", "Somewhat easy", "Difficult"]},

    {"q": "When answering, what approach do you follow?",
     "options": ["Think deeply before answering", "Answer with some thought", "Answer without much thinking"]},

    {"q": "How do you react when unsure about a question?",
     "options": ["Take time to analyze", "Make a quick guess", "Skip or avoid"]},

    {"q": "What best describes your current mood?",
     "options": ["Relaxed", "Balanced", "Tense"]},

    {"q": "How emotionally stable do you feel right now?",
     "options": ["Very stable", "Somewhat stable", "Unstable"]}
]

if not st.session_state.finished:
    q = tasks[st.session_state.q_idx]

    if st.session_state.q_start_time is None:
        st.session_state.q_start_time = time.time()

    st.write(f"**Q{st.session_state.q_idx+1}: {q['q']}**")

    selected = st.radio("Select your answer", q["options"],
                        key=f"q_{st.session_state.q_idx}")

    if st.button("Next"):
        rt = time.time() - st.session_state.q_start_time

        st.session_state.logs.append({
            "response": selected,
            "rt": rt
        })

        st.session_state.q_idx += 1
        st.session_state.q_start_time = time.time()

        if st.session_state.q_idx >= len(tasks):
            st.session_state.finished = True

        st.rerun()

# ===============================
# STEP 3: BALANCED RISK MODEL
# ===============================
if st.session_state.finished:
    st.subheader("🔬 Step 3: Risk Computation")

    logs = st.session_state.logs

    total_score = 0
    count = 0

    for l in logs:
        r = l["response"]
        score = 0.5

        if r in ["Take a break", "Doing nothing", "Scrolling casually"]:
            score += 0.15
        elif r in ["Watch something relaxing", "Enjoying entertainment"]:
            score += 0.08
        elif r in ["Engage in a challenging task", "Analyzing something"]:
            score -= 0.10

        if r in ["Uncertain", "Low trust"]:
            score += 0.15
        elif r in ["Partially certain", "Moderate trust"]:
            score += 0.05
        elif r in ["Completely certain", "Strong trust"]:
            score -= 0.10

        if r in ["Highly demanding", "Difficult"]:
            score += 0.10
        elif r in ["Moderately demanding", "Somewhat easy"]:
            score += 0.04
        elif r in ["Very easy", "Not demanding"]:
            score -= 0.08

        if r in ["Answer without much thinking", "Make a quick guess"]:
            score += 0.10
        elif r in ["Answer with some thought"]:
            score += 0.04
        elif r in ["Think deeply before answering", "Take time to analyze"]:
            score -= 0.08

        if r in ["Tense", "Unstable"]:
            score += 0.12
        elif r in ["Balanced", "Somewhat stable"]:
            score += 0.04
        elif r in ["Relaxed", "Very stable"]:
            score -= 0.10

        total_score += score
        count += 1

    avg_score = total_score / count

    rt_mean = np.mean([l["rt"] for l in logs])
    avg_score += (rt_mean / 60) * 0.05

    risk = np.clip(avg_score, 0, 1)

    st.metric("Final Risk Score", f"{risk:.1%}")

    # ===============================
    # 📊 RISK INTERPRETATION TABLE
    # ===============================
    st.markdown("### 📊 Risk Interpretation Table")

    risk_table = [
        ["0–20%", "😊 Happy", "Low Risk"],
        ["21–40%", "😐 Neutral", "Moderate"],
        ["41–60%", "😟 Slightly Stressed", "Medium"],
        ["61–80%", "😨 Fear / Stress", "High"],
        ["81–100%", "😡 Severe Stress", "Very High"]
    ]

    st.table({
        "Risk Range": [row[0] for row in risk_table],
        "Emotion": [row[1] for row in risk_table],
        "Level": [row[2] for row in risk_table]
    })

    # ===============================
    # 🎯 CATEGORY
    # ===============================
    risk_percent = risk * 100

    if risk_percent <= 20:
        category = "😊 Happy (Low Risk)"
    elif risk_percent <= 40:
        category = "😐 Neutral (Moderate)"
    elif risk_percent <= 60:
        category = "😟 Slightly Stressed (Medium)"
    elif risk_percent <= 80:
        category = "😨 High Stress (High Risk)"
    else:
        category = "😡 Severe Stress (Very High Risk)"

    st.markdown(f"### 🎯 Your Category: **{category}**")

    # ===============================
    # EMOTION DISPLAY
    # ===============================
    st.markdown(f"### 🎯 Detected Emotion: **{latest_emotion.capitalize()}**")

    # ===============================
    # 🧠 EXPLANATION (ADD HERE)
    # ===============================
    st.markdown("### 🧠 Explanation of Result")

    if risk < 0.3:
        st.success(
            "The user demonstrates stable cognitive behavior with confident decision-making, "
            "low mental stress, and a balanced emotional state."
        )

    elif risk < 0.6:
        st.warning(
            "The user shows moderate variation in performance with occasional uncertainty, "
            "slightly increased cognitive load, and mixed behavioral patterns."
        )

    else:
        st.error(
            "The user exhibits signs of cognitive stress, including low confidence, "
            "impulsive responses, and emotionally unstable behavior."
        )

    st.success("🎉 Screening complete")