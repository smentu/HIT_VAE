import matplotlib.pyplot as plt
import numpy as np


def visualize_samples(iter):
    images, labels = iter.next()

    n_samples = np.min((20, len(images)))

    fig = plt.figure(figsize=(15, 10))

    for idx in np.arange(n_samples):
        ax = fig.add_subplot(4, n_samples / 4, idx + 1, xticks=[], yticks=[])

        img = images[idx]
        ax.imshow(np.transpose(img, (1, 2, 0)).view(28, 28), cmap='gray')


def visualize_latent_1D(Z, labels, ylabel, clabel, savepath=None, colormap=None):

    fig, ax = plt.subplots(figsize=(10, 10))

    x = labels[ylabel]
    y = Z[:, 0]
    clabels = labels[clabel]

    if colormap:
        cmap = colormap
    else:
        cmap = 'Set1'

    scatter = ax.scatter(x, y, c=clabels, cmap=cmap)

    ax.legend(*scatter.legend_elements(), loc="upper right")

    ax.set_xlabel('latent')
    ax.set_ylabel(ylabel)
    ax.set_ylabel(ylabel)

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()


def visualize_latent(Z, labels, label_name, savepath=None, colormap=None):

    fig, ax = plt.subplots(figsize=(10, 10))

    x = Z[:, 0]
    y = Z[:, 1]

    if colormap:
        cmap = colormap
    else:
        cmap = 'Set1'

    clabels = labels[label_name]

    scatter = ax.scatter(x, y, c=clabels, cmap=cmap)

    ax.legend(*scatter.legend_elements(), loc="upper right")

    #plt.xticks([])
    #plt.yticks([])

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()
