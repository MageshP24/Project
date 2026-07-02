# Execution-Controlled Emotion-Conditioned Cognitive Risk Screening System

> **An Emotion-Guided, Explainable, and Safety-Oriented Cognitive Screening Framework with Supervisory Execution Control**

---

## 📌 Overview

The **Execution-Controlled Emotion-Conditioned Cognitive Risk Screening System** is an AI-driven framework designed to perform safe, explainable, and context-aware cognitive risk screening by integrating real-time affective analysis with behavioral interaction assessment.

Unlike conventional cognitive screening systems that treat emotion merely as an input feature, the proposed framework utilizes **emotional stability as a supervisory runtime control signal** that dynamically governs whether cognitive screening should be activated, restricted, or suspended.

The system continuously monitors a user's emotional state through live facial input, computes emotional stability, and authorizes screening only under validated stable conditions. This architecture prevents emotionally unstable conditions from generating misleading cognitive risk predictions and significantly reduces false-positive assessments.

---

## 🎯 Motivation

Traditional digital cognitive screening systems primarily rely on behavioral indicators such as:

- Response Accuracy
- Response Time
- Hint Usage
- Interaction Patterns

However, these systems fail to consider that temporary emotional disturbances such as:

- Stress
- Anxiety
- Frustration
- Anger
- Emotional Fatigue

can significantly alter user behavior.

As a result, emotion-induced behavioral variations are often incorrectly interpreted as cognitive deficits, leading to unreliable assessments.

This project addresses these limitations by introducing an emotion-guided execution-control architecture that ensures screening occurs only under emotionally reliable conditions.

---

## 🚀 Key Features

- ✅ Real-time facial emotion detection
- ✅ Emotional Stability Index (ESI) computation
- ✅ Multi-state execution-control architecture
- ✅ Emotion-guided screening authorization
- ✅ Session-adaptive behavioral reinterpretation
- ✅ Pre-computation pathway suppression
- ✅ State-adaptive bounded cognitive risk estimation
- ✅ Intrinsic execution-trace explainability
- ✅ Screening safety control mechanisms
- ✅ Human review triggering under prolonged instability
- ✅ Real-time deployment on standard consumer devices

---

# 🏗️ System Architecture

The proposed framework consists of the following major modules:

### 1. Affective Signal Processing Module

- Captures live facial video streams using a camera sensor.
- Performs real-time CNN-based emotion classification.
- Generates emotion probability vectors.
- Produces structured affective context signals.

### 2. Emotional Stability Evaluation Module

- Computes the Emotional Stability Index (ESI) over consecutive emotional observations.
- Evaluates emotional consistency and reliability.
- Uses emotional stability as a supervisory runtime control parameter.

### Emotional Stability Index (ESI)

```text
ESI = 1 - ((1/T) × Σ Var(Pt))
```

Where:

- `Pt` = Emotion probability vector at time `t`
- `Var(Pt)` = Variance of emotion probabilities
- `T` = Temporal observation window

### 3. Supervisory Execution-State Control Engine

The system employs a deterministic finite-state control architecture consisting of five execution states.

| State | Description |
|--------|-------------|
| S0 | Emotion Unavailable |
| S1 | Emotion Unstable |
| S2 | Emotion Stabilizing |
| S3 | Emotion Stable |
| S4 | Screening Cooldown |

Only **S3 (Emotion Stable)** authorizes cognitive screening.

The engine incorporates:

- Stability Threshold (θ)
- Dwell Time (τ)
- Stability Persistence Window (T)
- Cooldown Timer (Tc)
- Hysteresis Control

---

### 4. Behavioral Feature Capture Module

Behavioral data is collected only during authorized screening sessions.

Captured features include:

- Response Accuracy
- Response Latency
- Hint Usage
- Interaction Consistency

---

### 5. Session-Adaptive Emotion-Conditioned Behavioral Transformation Module

Behavioral interaction features are transformed based on:

- Current execution state
- Session-level emotional history
- Emotional stability

Behavioral transformation:

```text
B' = Ms × B
```

Where:

- `B` = Behavioral feature vector
- `Ms` = Execution-state transformation matrix
- `B'` = Emotion-conditioned behavioral representation

The transformation matrix is dynamically recalibrated throughout the session to account for prolonged emotional instability.

---

### 6. State-Adaptive Bounded Risk Regulation Module

This module computes a screening-level cognitive risk indicator while enforcing operational safety constraints.

Risk computation:

```text
R = min(Rmax, (w1 × A) + (w2 × RT) + (w3 × H))
```

Where:

- `A` = Response Accuracy
- `RT` = Response Time
- `H` = Hint Usage
- `w1, w2, w3` = State-dependent weights
- `Rmax` = Maximum allowable risk bound

---

### 7. Intrinsic Explanation Module

Unlike conventional post-hoc explainability approaches, the proposed framework generates explanations intrinsically during runtime.

Generated explanations include:

- Emotional stability status
- Execution-state transitions
- Behavioral transformation rationale
- Risk computation trace
- Safety-control decisions

---

### 8. Screening Safety Control Module

The safety controller performs three independent functions:

#### Risk Escalation Regulation

Prevents sudden spikes in cognitive risk under unstable emotional conditions.

#### Screening Suspension Mechanism

Automatically suspends screening when emotional instability persists.

#### Human Review Trigger

Flags highly unstable sessions for human review.

---

# ⚙️ Working Principle

The system operates according to the following workflow:

1. Capture live facial input through a camera sensor.
2. Perform real-time emotion classification.
3. Compute Emotional Stability Index (ESI).
4. Determine the current execution state (S0–S4).
5. If emotional stability is validated (**S3**):
   - Present cognitive tasks.
   - Capture behavioral responses.
   - Apply session-adaptive behavioral transformation.
   - Estimate bounded cognitive risk.
   - Generate intrinsic explanations.
6. If emotional stability is not validated:
   - Restrict or suspend screening.
   - Display a suspension notification.
7. Display:
   - Cognitive risk score
   - Execution state
   - Explanation trace

---

## 📂 Project Structure

```text
Execution-Controlled-Cognitive-Risk-Screening-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── emotion_model.pth
│   └── risk_model.pkl
│
├── src/
│   ├── emotion_detection.py
│   ├── emotional_stability.py
│   ├── execution_controller.py
│   ├── behavioral_transformation.py
│   ├── risk_estimation.py
│   ├── safety_controller.py
│   └── explanation_generator.py
│
├── data/
├── notebooks/
├── results/
├── images/
│   ├── system_architecture.png
│   └── working_principle.png
│
└── docs/
```

---

## 🛠️ Technology Stack

### Programming Language

- Python

### Deep Learning

- PyTorch
- Torchvision

### Computer Vision

- OpenCV

### Machine Learning

- Scikit-learn

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### User Interface

- Streamlit

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Execution-Controlled-Cognitive-Risk-Screening-System.git

cd Execution-Controlled-Cognitive-Risk-Screening-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📸 Screenshots

### System Architecture

```markdown
![System Architecture](images/system_architecture.png)
```

### Working Principle

```markdown
![Working Principle](images/working_principle.png)
```

---

## 🔬 Applications

- Early-stage cognitive screening
- Educational analytics
- Emotion-aware intelligent tutoring systems
- Learning disability screening support
- Human-centered AI applications
- Remote behavioral assessment
- Context-aware educational technologies

---

## 🏆 Technical Contributions

- Supervisory emotion-driven execution control
- Multi-state runtime governance framework
- Temporal emotional stability quantification
- Session-adaptive behavioral reinterpretation
- Pre-computation screening pathway suppression
- State-adaptive bounded risk regulation
- Intrinsic execution-trace explainability
- Safety-oriented screening control

---

## 📄 Patent Information

**Patent Title:**

**Execution-Controlled Emotion-Conditioned Cognitive Risk Screening System with Session-Adaptive Behavioral Interpretation and Intrinsic Explainability**

**Status:** Patent Disclosure Submitted / Under Review

---

## ⚠️ Disclaimer

This framework is intended solely for:

- Screening
- Educational support
- Decision assistance

This system **does not provide medical diagnosis, clinical diagnosis, or treatment recommendations**.

Any screening outcome should be interpreted only as a non-diagnostic risk indicator.

---

## 🔮 Future Work

- Multimodal affective sensing
- Speech-based emotion analysis
- Longitudinal cognitive monitoring
- Personalized intervention recommendation
- Edge-device deployment
- Federated learning integration
- Adaptive curriculum recommendation

---

## 📚 Citation

```bibtex
@misc{magesh2026executioncontrolled,
  title={Execution-Controlled Emotion-Conditioned Cognitive Risk Screening System with Session-Adaptive Behavioral Interpretation and Intrinsic Explainability},
  author={Magesh P},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Magesh P**

Department of Computer Science and Engineering  
Vellore Institute of Technology (VIT), Vellore, India

---

⭐ If you find this project useful, please consider giving it a star.
