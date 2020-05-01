import numpy as np

DATA_PATH = '../../data/'
GROUND_TRUTH_PATH = DATA_PATH + 'cross_validation/labels/'
PREDICTED_PATH = DATA_PATH + 'predictions/'
NUM_FOLDS = 5

MODELS = ['shortest_realworld_distance', 'most_common_transition', 'most_similar_trajectory', 'hand_crafted_features', 'fully_connected', 'lstm', 'gru']

for model in MODELS:
    top_1s = []
    top_3s = []
    for fold_num in range(1, NUM_FOLDS + 1):
        fold = str(fold_num)
        ground_truth_labels = np.load(GROUND_TRUTH_PATH + 'test_fold' + fold + '.npy').astype(int)
        predictions = np.load(PREDICTED_PATH + model + '/fold_' + fold + '.npy').astype(int)
        correct_top_1 = 0
        correct_top_3 = 0
        for row in range(len(ground_truth_labels)):
            ground_truth = ground_truth_labels[row]
            top_1 = predictions[row][0:1]
            top_3 = predictions[row][0:3]
            if ground_truth in top_1:
                correct_top_1 += 1
            if ground_truth in top_3:
                correct_top_3 += 1
        top_1s.append(correct_top_1 / len(ground_truth_labels))
        top_3s.append(correct_top_3 / len(ground_truth_labels))
    print()
    print(model)
    print('Average top 1: ', np.round(np.mean(top_1s) * 100, 1))
    print('Average top 3: ', np.round(np.mean(top_3s) * 100, 1))
print()
