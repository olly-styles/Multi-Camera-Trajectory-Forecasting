import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = '../../data/'
NUM_CAMERAS = 15
NUM_FOLDS = 5
OFFSET_LEN = 10

for fold_num in range(1, NUM_FOLDS + 1):
    print('Plotting for fold ', fold_num)
    transition_matrix = np.zeros((NUM_CAMERAS, NUM_CAMERAS), dtype='uint16')
    fold = str(fold_num)

    train_input_cameras = np.load(DATA_PATH + 'cross_validation/input_cameras/train_fold' + fold + '.npy')
    train_labels = np.load(DATA_PATH + 'cross_validation/labels/train_fold' + fold + '.npy')

    for camera in range(1, NUM_CAMERAS + 1):
        camera_labels = train_labels[train_input_cameras == camera]
        unique, counts = np.unique(camera_labels, return_counts=True)
        transition_matrix[camera - 1, unique - 1] = counts

    # OFFSET_LEN results per transition. Divide by this to get
    # number of transitions
    transition_matrix = transition_matrix / OFFSET_LEN
    transition_matrix = transition_matrix.astype('uint16')

    plt.clf()
    fig, ax = plt.subplots()
    ax = sns.heatmap(transition_matrix, cmap=sns.color_palette("RdBu_r", transition_matrix.max()))
    ax.set_xticklabels(np.arange(1, 16), rotation=0)
    ax.set_yticklabels(np.arange(1, 16), rotation=0)
    ax.set_xlabel("Departure camera", fontsize=18)
    ax.set_ylabel("Entrance camera", fontsize=18)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig('./figures/transition_frequency_fold_' + fold + '.png')
