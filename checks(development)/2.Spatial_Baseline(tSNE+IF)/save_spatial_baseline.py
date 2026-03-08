# This file is to save the results of our baseline spatial branch evaluation on a small test set of 20 images (10 real from CelebA and 10 fake from StyleGAN / thispersondoesnotexist.com).
# We will save a CSV with the image paths, true labels, anomaly scores, and decisions (REAL/FAKE based on a 0.5 threshold). 
# We will also save a text summary of the results and analysis. This serves as a sanity check to ensure our spatial branch

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
import os
from datetime import datetime

# Your results
data = {
    "image": [
        # ---Real Images-------------
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000001.jpg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000002.jpg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000003.jpg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000004.jpg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000005.jpg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000006.jpg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000007.jpg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000008.jpg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000009.jpg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000010.jpg",

        # ---Fake Images-------------
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-1.jpeg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-2.jpeg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-3.jpeg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-4.jpeg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-5.jpeg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-6.jpeg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-7.jpeg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-8.jpeg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-9.jpeg",
            "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-10.jpeg",
    ],
    "true_label": ["REAL"] * 10 + ["FAKE"] * 10,
    "anomaly_score": [
        0.696, 0.249, 0.628, 0.221, 0.170,
        0.172, 0.700, 0.315, 0.658, 0.103,
        0.000, 0.246, 0.155, 0.309, 0.871,
        1.000, 0.041, 0.284, 0.559, 0.233
    ]
}

df = pd.DataFrame(data)
df["decision"] = df["anomaly_score"].apply(lambda x: "FAKE" if x > 0.5 else "REAL")
df["correct"]  = df["true_label"] == df["decision"]

# Save CSV
output_dir = "outputs/evaluation/baseline_spatial_only"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "baseline_results.csv"), index=False)

# Compute metrics
real_df = df[df["true_label"] == "REAL"]
fake_df = df[df["true_label"] == "FAKE"]

total_correct    = df["correct"].sum()
real_correct     = real_df["correct"].sum()
fake_correct     = fake_df["correct"].sum()
overall_accuracy = total_correct / len(df)
mean_real_score  = real_df["anomaly_score"].mean()
mean_fake_score  = fake_df["anomaly_score"].mean()

# Save summary
summary = f"""
BASELINE EVALUATION — SPATIAL BRANCH ONLY
==========================================
Date:             {datetime.now().strftime("%Y-%m-%d %H:%M")}
Model:            DINO ViT-S/8 + Isolation Forest
Training data:    CelebA 10k real faces
Test set:         10 real (CelebA) + 10 fake (StyleGAN / thispersondoesnotexist.com)
Threshold:        0.5

RESULTS
──────────────────────────────────────────
Overall accuracy: {overall_accuracy:.1%}  ({total_correct}/20 correct)
Real faces:       {real_correct}/10 correctly identified
Fake faces:       {fake_correct}/10 correctly identified

Mean real score:  {mean_real_score:.3f}
Mean fake score:  {mean_fake_score:.3f}
Score separation: {mean_fake_score - mean_real_score:.3f}  (positive = good, negative = bad)

ANALYSIS
──────────────────────────────────────────
- Spatial branch alone is insufficient against StyleGAN generated faces
- StyleGAN faces are semantically convincing at the embedding level
- DINO embeddings + Isolation Forest cannot distinguish high quality
  synthetic faces from real faces reliably at this training scale
- Mean fake score LOWER than mean real score indicates the model
  is not yet separating distributions correctly

WHY THIS IS EXPECTED
──────────────────────────────────────────
1. StyleGAN is the hardest possible test — generates photorealistic faces
2. Only 10k training images — manifold boundary is not tight enough
3. Spatial embeddings alone miss pixel level artifacts
4. No temporal signal yet — vPPG branch not integrated

WHAT WILL IMPROVE THIS
──────────────────────────────────────────
1. Autoencoder — adds pixel level reconstruction error signal
2. Full 160k CelebA training — tighter real face manifold
3. FaceForensics++ test set — face swaps have clearer artifacts
4. Temporal branch (vPPG) — physiological signal catches StyleGAN fakes
5. Fusion of all signals — no single branch needs to be perfect
"""

with open(os.path.join(output_dir, "baseline_summary.txt"), "w") as f:
    f.write(summary)

print(summary)
print(f"Saved to: {output_dir}")