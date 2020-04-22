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


shortest_real_world = np.load(DATA_PATH + 'real_world_distance/shortest_real_world_distance.npy')

for fold_num in range(1, NUM_FOLDS + 1):
    print('Computing for fold ', fold_num)
    fold = str(fold_num)

    # Get input cameras
    test_input_cameras = np.load(DATA_PATH + 'cross_validation/input_cameras/test_fold' + fold + '.npy')

    # Get labels
    test_labels = np.load(DATA_PATH + 'cross_validation/labels/test_fold' + fold + '.npy')

    all_predictions = np.zeros((len(test_labels), NUM_CAMERAS), dtype='uint8')
    for camera in range(1, NUM_CAMERAS + 1):

        ranked_camera_predictions = shortest_real_world[camera - 1]
        ranked_camera_predictions = append_missing_cameras(ranked_camera_predictions, NUM_CAMERAS)
        camera_indexes = np.argwhere(test_input_cameras == camera).squeeze()
        all_predictions[camera_indexes] = np.repeat(ranked_camera_predictions[np.newaxis, :], len(camera_indexes), axis=0)

    # All predictions should be assigned a camera
    assert np.sum(all_predictions == 0) == 0
    np.save(DATA_PATH + 'predictions/shortest_realworld_distance/fold_' + fold + '.npy', all_predictions)
