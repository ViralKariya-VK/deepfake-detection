import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def load_embeddings(embeddings_dir):
    embeddings = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
    filenames = np.load(os.path.join(embeddings_dir, "filenames.npy"))

    print(f"Loaded embeddings shape: {embeddings.shape}")
    return embeddings, filenames


def run_tsne(embeddings, perplexity=30, random_state=42):
    print("Running t-SNE")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )

    reduced = tsne.fit_transform(embeddings)
    
    return reduced

    '''
    `n_components=2` — compress to 2D for visualization

    `perplexity=30` — roughly means "consider 30 nearest neighbors when arranging points". Rule of thumb:

    500  images  → perplexity 20-30
    5000 images  → perplexity 30-50
    10k+ images  → perplexity 50
    `'''

def plot_tsne(reduced, labels=None, save_path=None):
    plt.figure(figsize=(10, 8))

    if labels is None:
        # Single color - all real faces

        plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c='steelblue',
            alpha=0.5,
            s=10
        )

        plt.title("t-SNE – Real Face Embeddings (DINO ViT)")
    else:
        # Two colors - real vs fake (Phase 3 onwards)
        colors = ['steelblue' if l == 0 else 'crimson' for l in labels]
        plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=colors,
            alpha=0.5,
            s=10
        )
        plt.scatter("t-SNE – Real (blue) vs Fake (red)")

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()


def visualize_embeddings(embeddings_dir, output_dir, perplexity=30):
    embeddings, filenames = load_embeddings(embeddings_dir)

    # t-SNE works best on a sample for large datasets
    # For 10k+ subsamples 5000 to keep it fast

    if len(embeddings) > 100000:
        idx = np.random.choice(len(embeddings), 100000, replace=False)
        embeddings = embeddings[idx]
        print(f"Subsampled to 5000 embeddings for t-SNE")

    reduced = run_tsne(embeddings, perplexity=perplexity)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "tsne_real_faces.png")
    plot_tsne(reduced, save_path=save_path)

    # Also save the reduced coordinates for later use
    np.save(os.path.join(output_dir, "tsne_coords.npy"), reduced)


    