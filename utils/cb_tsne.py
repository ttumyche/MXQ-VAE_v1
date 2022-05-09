# for codebook tsne
# plotting each idx in codebook

import os
import wandb
from glob import glob
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from utils import set_seed

def codebook_tsne(epoch, model):
    print('Visualize codebook t-SNE')

    tsne = TSNE(n_components=2, learning_rate=100, perplexity=15, random_state=0)
    latent_vectors = model.module.codebook.state_dict()['weight'].detach().cpu().numpy()
    embed_vectors = tsne.fit_transform(latent_vectors)
    xs, ys = embed_vectors[:, 0], embed_vectors[:, 1]

    plt.scatter(xs, ys)

    plt.xlim(xs.min(), xs.max())
    plt.ylim(ys.min(), ys.max())
    plt.xlabel('X axis', labelpad=10)
    plt.ylabel('Y axis', labelpad=10)

    wandb.log({'codebook': plt}, step=epoch)