import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

LINK_AIST = [
    (15, 14), (14, 12),
    (16, 13), (13, 11),
    (12, 11), (12, 6), (6, 5), (5, 11),
    (6, 8), (8, 10), 
    (5, 7), (7, 9),
    (4, 2), (2, 0), (0, 1), (1, 3)
]

LINK_MOTORICA = [
    (0, 1), (0, 2), (0, 3),
    (2, 5), (5, 10), (10, 15),
    (3, 6), (6, 11), (11, 16),
    (1, 4), (4, 7), (7, 12),
    (4, 8), (8, 13), (13, 17), (17, 19),
    (4, 9), (9, 14), (14, 18), (18, 20) 
]

LINK_SMPL = [
    (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 14), (9, 13), (9, 12), (12, 15),
    (2, 5), (5, 8), (8, 11),
    (1, 4), (4, 7), (7, 10),
    (14, 17), (17, 19), (19, 21), (21, 23),
    (13, 16), (16, 18), (18, 20), (20, 22)
]

def choose_majority(attributes):
    if isinstance(attributes, torch.Tensor):
        attributes = attributes.cpu().numpy()
    
    majority_att = []
    for i in range(attributes.shape[0]):
        att = attributes[i]
        bin = np.bincount(att)
        max_value = np.argmax(bin)
        majority_att.append(max_value)

    return np.array(majority_att)

def compute_tsne(latents, labels=None, path=None):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    latents_tsne = tsne.fit_transform(latents)

    if path is not None:
        # ========== PLOT ==========
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.title('t-SNE of VAE Latent Space')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        if labels is not None:
            plt.legend(*scatter.legend_elements(), title="Labels", loc="best", fontsize=8)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(path)
