import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# script for processing MNIST images into a form that is easily fed into torch models

digit_counts = {'1': 100, '2': 100, '3': 100, '4': 100, '5': 100, '6': 100, '7': 100, '8': 100, '9': 100}

source_location = '../../data/mnist_data/trainingSet'
generated_destination_dir = '../../mnist_gen'

if not os.path.exists(generated_destination_dir):
    os.mkdir(generated_destination_dir)

sample_index = 0
label_dict = {}

for digit in digit_counts.keys():
    data_path = os.path.join(source_location, digit)
    files = glob.glob('{}/*.jpg'.format(data_path))

    for i in range(digit_counts[digit]):
        img = plt.imread(files[i])

        label_dict[sample_index] = digit

        fig = plt.figure(frameon=False, dpi=28)
        fig.set_size_inches(1, 1)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, cmap='gray')
        plt.axis('off')

        filename = '{}.png'.format(sample_index)
        plt.savefig(os.path.join(generated_destination_dir, filename))

        plt.close()

        sample_index += 1

# print(label_dict)

df = pd.DataFrame.from_dict(label_dict, orient='index', columns=['digit'])

df.to_csv(os.path.join(generated_destination_dir, 'labels.csv'))
