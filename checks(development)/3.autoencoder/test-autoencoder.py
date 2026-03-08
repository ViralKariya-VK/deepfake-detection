import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch

from spatial_branch.autoenconder import AutoencoderTrainer

    
# ── Config ────────────────────────────────────────────────
REAL_PATHS = [
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
]

FAKE_PATHS = [
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
]

OUTPUT_DIR = "outputs/evaluation/autoencoder_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────
trainer = AutoencoderTrainer(
    images_dir     = "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/processed/celeba_aligned",
    checkpoint_dir = "checkpoints/"
)
trainer.load("autoencoder_final.pth")

# ── Score all images ──────────────────────────────────────
all_paths  = REAL_PATHS + FAKE_PATHS
all_labels = ["REAL"] * len(REAL_PATHS) + ["FAKE"] * len(FAKE_PATHS)

scores   = []
heatmaps = []

print(f"\n{'Image':<25} {'Label':<8} {'Recon Error'}")
print("─" * 45)

for path, label in zip(all_paths, all_labels):
    score, heatmap = trainer.get_reconstruction_error(path)
    scores.append(score)
    heatmaps.append(heatmap)
    print(f"{os.path.basename(path):<25} {label:<8} {score:.6f}")

scores      = np.array(scores)
real_scores = scores[:len(REAL_PATHS)]
fake_scores = scores[len(REAL_PATHS):]

print(f"\n{'─' * 45}")
print(f"Mean real reconstruction error: {real_scores.mean():.6f}")
print(f"Mean fake reconstruction error: {fake_scores.mean():.6f}")
print(f"Separation:                     {fake_scores.mean() - real_scores.mean():.6f}")
print(f"{'─' * 45}\n")

# ── Heatmap grid ──────────────────────────────────────────
n   = len(all_paths)
fig = plt.figure(figsize=(15, n * 3))
gs  = gridspec.GridSpec(n, 3, figure=fig)

for i, (path, label, score, heatmap) in enumerate(
    zip(all_paths, all_labels, scores, heatmaps)
):
    original  = Image.open(path).convert("RGB").resize((224, 224))
    tensor    = trainer.transform(original).unsqueeze(0).to(trainer.device)

    with torch.no_grad():
        reconstructed = trainer.model(tensor)

    recon_img = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_img = (recon_img * 255).astype("uint8")

    ax1 = fig.add_subplot(gs[i, 0])
    ax1.imshow(original)
    ax1.set_title(f"{label} — {os.path.basename(path)}", fontsize=8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[i, 1])
    ax2.imshow(recon_img)
    ax2.set_title("Reconstructed", fontsize=8)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[i, 2])
    im  = ax3.imshow(heatmap, cmap="hot", interpolation="nearest")
    ax3.set_title(f"Error: {score:.6f}", fontsize=8)
    ax3.axis("off")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

plt.suptitle("Autoencoder Reconstruction Error — Real vs Fake",
              fontsize=12, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "heatmap_grid.png")
plt.savefig(output_path, dpi=120, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.show()
