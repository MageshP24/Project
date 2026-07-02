import streamlit as st
import numpy as np

st.set_page_config(page_title="Emotion-Guided Cognitive Risk", layout="centered")

st.title("🧠 Emotion-Guided Cognitive Risk Screening System")
st.caption("Educational screening • Not a medical diagnosis")

st.divider()

# -------------------------
# INPUTS (simulate live system)
# -------------------------
st.subheader("Student Session Inputs")

emotion_level = st.slider(
    "Frustration Level (from emotion CNN)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05
)

accuracy = st.slider(
    "Task Accuracy",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05
)

response_time = st.slider(
    "Average Response Time (seconds)",
    min_value=5,
    max_value=60,
    value=25
)

st.divider()

# -------------------------
# BASE COGNITIVE RISK (simulated RF output)
# -------------------------
# This mimics your trained Dyslexia RF distribution
base_risk = np.clip(
    0.15 + (1 - accuracy) * 0.4 + (response_time / 60) * 0.15,
    0, 1
)

# -------------------------
# EMOTION-GUIDED FUSION (YOUR NOVEL CORE)
# -------------------------
if emotion_level >= 0.6:
    gated_risk = base_risk * 0.4
    gate_msg = "⚠️ High frustration detected → cognitive risk down-weighted"
else:
    gated_risk = base_risk
    gate_msg = "✅ Emotional state stable → full cognitive confidence"

# -------------------------
# OUTPUTS
# -------------------------
st.subheader("Risk Assessment")

col1, col2 = st.columns(2)
with col1:
    st.metric("Base Cognitive Risk", f"{base_risk:.1%}")

with col2:
    st.metric("Emotion-Gated Risk", f"{gated_risk:.1%}")

st.info(gate_msg)

st.divider()

# -------------------------
# EXPLANATION (XAI for viva)
# -------------------------
st.subheader("Explanation (XAI)")

st.markdown(
"""
- **Base risk** is estimated from learning behaviour patterns  
- **Emotion is treated as a transient state**, not a disorder  
- During high frustration, the system **suppresses risk confidence**
- This prevents **false positive cognitive risk alerts**
"""
)

st.success("🎉 Emotion-guided screening complete")

st.caption("For research and educational assistance only.")
