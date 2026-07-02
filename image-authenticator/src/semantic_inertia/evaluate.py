import os
import random
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

from .transformations import get_transformations
from .semantic_encoder import extract_embedding
from .inertia_score import semantic_inertia_features


# --------------------------------------------------
# Helper: sample images safely
# --------------------------------------------------
def sample_images(folder_path, max_images):
    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    if len(images) > max_images:
        images = random.sample(images, max_images)
    return images


# --------------------------------------------------
# Feature extraction per folder
# --------------------------------------------------
def process_folder(path, max_images):
    features = []
    image_list = sample_images(path, max_images)
    total = len(image_list)

    print(f"\nProcessing {total} images from: {path}")

    for idx, img_name in enumerate(image_list, start=1):
        image = Image.open(os.path.join(path, img_name)).convert("RGB")

        embeddings = []
        images_tensor = []

        for t in get_transformations():
            img_t = t(image)  # Tensor (C, H, W)
            images_tensor.append(img_t)

            img_pil = Image.fromarray(
                (img_t.permute(1, 2, 0).numpy() * 255).astype("uint8")
            )
            embeddings.append(extract_embedding(img_pil))

        # 🔥 Semantic inertia features (MATCHES inertia_score.py)
        mean_d, var_d, max_d, drift_energy, freq_instability = (
            semantic_inertia_features(embeddings, images_tensor)
        )

        features.append([
            mean_d,
            var_d,
            max_d,
            drift_energy,
            freq_instability
        ])

        if idx % 10 == 0 or idx == total:
            percent = (idx / total) * 100
            print(f"Completed: {percent:.2f}% ({idx}/{total})", end="\r")

    print("\nCompleted processing:", path)
    return features


# --------------------------------------------------
# Paths & parameters
# --------------------------------------------------
REAL_PATH = "data/images/train/real"
FAKE_PATH = "data/images/train/ai"
MAX_IMAGES = 10000   # increase later if system allows


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------
real_features = process_folder(REAL_PATH, MAX_IMAGES)
fake_features = process_folder(FAKE_PATH, MAX_IMAGES)

X = np.array(real_features + fake_features)
y = np.array([1] * len(real_features) + [0] * len(fake_features))


# --------------------------------------------------
# 🔥 Feature normalization (CRITICAL for SVM)
# --------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)


# --------------------------------------------------
# Train / Test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# --------------------------------------------------
# Train SVM (RBF kernel)
# --------------------------------------------------
clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 🔥 Optimal threshold (Youden’s J)
# --------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, probs)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]

preds = (probs >= best_threshold).astype(int)


# --------------------------------------------------
# Metrics
# --------------------------------------------------
accuracy  = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall    = recall_score(y_test, preds)
f1        = f1_score(y_test, preds)
roc_auc   = roc_auc_score(y_test, probs)

print("\nEvaluation Results (Semantic Inertia + SVM)")
print("------------------------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)
print("ROC-AUC  :", roc_auc)


# --------------------------------------------------
# Save results
# --------------------------------------------------
os.makedirs("results", exist_ok=True)

with open("results/semantic_inertia_metrics.txt", "w") as f:
    f.write("Semantic Inertia–Based Evaluation Results\n")
    f.write("----------------------------------------\n")
    f.write(f"Total Real Images Used     : {len(real_features)}\n")
    f.write(f"Total Synthetic Images Used: {len(fake_features)}\n\n")
    f.write("Classifier : SVM (RBF kernel)\n")
    f.write(
        "Features   : Mean Drift, Variance Drift, Max Drift, "
        "Drift Energy, Frequency Instability\n\n"
    )
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n")
    f.write(f"ROC-AUC   : {roc_auc:.4f}\n")

print("\n✅ Results saved to: results/semantic_inertia_metrics.txt")
