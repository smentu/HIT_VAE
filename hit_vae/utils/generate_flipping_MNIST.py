import os
import glob
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy import ndimage
import matplotlib.pyplot as plt

#digit_counts = {'7': 10, '9': 10, '4': 10}
digit_counts = {'3': 30}

source_location = '../../data/mnist_data/trainingSet'
generated_destination_dir = '../../data/gen/mnist_flipping'

if not os.path.exists(generated_destination_dir):
    os.mkdir(generated_destination_dir)

sickness_probability = 0.5
sample_index = 0
subject_index = 0
label_dict = {}

timestamps = np.arange(-10, 10)

for digit in digit_counts.keys():
    data_path = os.path.join(source_location, digit)
    files = glob.glob('{}/*.jpg'.format(data_path))

    for i in range(digit_counts[digit]):

        original = plt.imread(files[i])

        sick = np.random.binomial(1, sickness_probability)
        rotations = np.ones(len(timestamps)) + np.random.normal(0, 3, len(timestamps))

        if sick:
            print('subject {} is sick'.format(subject_index))
            #print(timestamps)
            #print(sigmoid(timestamps))
            rotations += 45 * sigmoid(timestamps)
            #print(rotations)
        else:
            print('subject {} is not sick'.format(subject_index))

        #print(rotations)

        for i, rotation in enumerate(rotations):

            img = ndimage.rotate(original, angle=rotation, reshape=False)

            # plt.imshow(img)
            #
            # plt.show()

            label_dict[sample_index] = [subject_index, digit, rotation, sick, timestamps[i]]

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

        subject_index += 1

#print(label_dict)

df = pd.DataFrame.from_dict(label_dict, orient='index', columns=['subject', 'digit', 'angle', 'sick', 'timestamp'])

df.to_csv(os.path.join(generated_destination_dir, 'labels.csv'))
