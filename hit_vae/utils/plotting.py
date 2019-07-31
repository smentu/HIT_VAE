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

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()

def visualize_latent_2D(Z, labels, savepath=None, colormap=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    x = Z[:, 0]
    y = Z[:, 1]

    if colormap:
        cmap = colormap
    else:
        cmap = 'Set1'

    clabels = labels
    scatter = ax.scatter(x, y, c=clabels, cmap=cmap)
    ax.legend(*scatter.legend_elements(), loc="upper right")

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()


def categorical_plot(x_data, y_data, categories, ax, tensors=False, savepath=None):
    if tensors:
        x_data = x_data.cpu().numpy()
        y_data = y_data.cpu().numpy()
        categories = categories.cpu().numpy()

    for c in np.unique(categories):
        indices = np.where(categories == c)

        x = x_data[indices]
        y = y_data[indices]

        order = np.argsort(x)

        ax.plot(x[order], y[order])

    if not savepath == None:
        plt.savefig(savepath)
