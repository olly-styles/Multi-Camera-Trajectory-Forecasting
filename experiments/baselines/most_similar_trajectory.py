import numpy as np

DATA_PATH = '../../data/'
NUM_CAMERAS = 15
NUM_FOLDS = 5


def append_missing_cameras(predictions, num_cameras):
    '''
    If any cameras are missing from the predictions, these are appended
    '''
    all_cameras = np.arange(1, num_cameras + 1)
    missing_cameras = np.setdiff1d(all_cameras, predictions)
    return np.append(predictions, missing_cameras)


for fold_num in range(1, NUM_FOLDS + 1):
    print('Computing for fold ', fold_num)
    fold = str(fold_num)

    # Get inputs for train and test
    train_inputs = np.load(DATA_PATH + 'cross_validation/inputs/train_fold' + fold + '.npy')
    test_inputs = np.load(DATA_PATH + 'cross_validation/inputs/test_fold' + fold + '.npy')
    train_inputs = train_inputs.reshape(-1, 40)
    test_inputs = test_inputs.reshape(-1, 40)

    # Get other data
    train_input_cameras = np.load(DATA_PATH + 'cross_validation/input_cameras/train_fold' + fold + '.npy')
    test_input_cameras = np.load(DATA_PATH + 'cross_validation/input_cameras/test_fold' + fold + '.npy')
    train_labels = np.load(DATA_PATH + 'cross_validation/labels/train_fold' + fold + '.npy')

    all_predictions = np.zeros((len(test_inputs), 15), dtype='uint8')
    for row in range(len(test_inputs)):
        test_input_camera = test_input_cameras[row]
        test_trajectory = test_inputs[row]
        cam_inputs = train_inputs[np.argwhere(train_input_cameras == test_input_camera)].squeeze()
        cam_labels = train_labels[np.argwhere(train_input_cameras == test_input_camera)].squeeze()

        idx = np.argsort(np.abs(cam_inputs - test_trajectory).sum(axis=1))
        sorted_predictions = cam_labels[idx]

        _, idx = np.unique(sorted_predictions, return_index=True)
        ranked_predictions = sorted_predictions[np.sort(idx)]

        ranked_predictions = append_missing_cameras(ranked_predictions, NUM_CAMERAS)
        all_predictions[row] = ranked_predictions
    np.save(DATA_PATH + 'predictions/most_similar_trajectory/fold_' + fold + '.npy', all_predictions)
