import numpy as np

DATA_PATH = '../data/'

all_inputs = np.zeros((0, 10, 4))
all_labels = np.zeros((0))
all_input_cameras = np.zeros((0))
all_days = np.zeros((0))


def split_array(array, train_indexs, val_indexs, test_indexs):
    train_data = array[train_indexs]
    val_data = array[val_indexs]
    test_data = array[test_indexs]
    return train_data, val_data, test_data


def save_data(save_path, fold, train_data, val_data, test_data):
    np.save(save_path + 'train_fold' + str(fold) + '.npy', train_data)
    np.save(save_path + 'val_fold' + str(fold) + '.npy', val_data)
    np.save(save_path + 'test_fold' + str(fold) + '.npy', test_data)


for day_num in range(1, 21):
    day = 'day' + str(day_num)

    # Read data
    inputs = np.load(DATA_PATH + 'inputs/inputs_' + day + '.npy')
    labels = np.load(DATA_PATH + 'labels/one_class_classification/one_class_classification_label_' + day + '.npy')
    input_cameras = np.load(DATA_PATH + 'input_cameras/input_cameras_' + day + '.npy')
    days = np.repeat(np.array([day_num]), len(labels))

    # Test all are same length
    assert len(days) == len(input_cameras) == len(labels) == len(inputs)

    # Append to create complete dataset
    all_inputs = np.append(all_inputs, inputs, axis=0)
    all_labels = np.append(all_labels, labels)
    all_input_cameras = np.append(all_input_cameras, input_cameras)
    all_days = np.append(all_days, days)

for fold in [1, 2, 3, 4, 5]:
    # Get indexes
    train_indexs = np.argwhere(all_days % 5 != fold - 1).flatten()
    test_indexs = np.argwhere(all_days % 5 == fold - 1).flatten()
    # Must be rounded to multiple of 10 to ensure same sample with different offset
    # does not appear in both val and test
    val_indexs = test_indexs[0:int(len(test_indexs) / 20) * 10]
    test_indexs = test_indexs[int(len(test_indexs) / 20) * 10:]

    assert len(np.intersect1d(train_indexs, val_indexs)) == 0
    assert len(np.intersect1d(val_indexs, test_indexs)) == 0
    assert len(np.intersect1d(train_indexs, test_indexs)) == 0

    print(len(train_indexs), len(val_indexs), len(test_indexs))

    train_inputs, val_inputs, test_inputs = split_array(all_inputs, train_indexs, val_indexs, test_indexs)
    train_labels, val_labels, test_labels = split_array(all_labels.astype('uint8'), train_indexs, val_indexs, test_indexs)
    train_input_cameras, val_input_cameras, test_input_cameras = split_array(all_input_cameras.astype('uint8'), train_indexs, val_indexs, test_indexs)

    save_data(DATA_PATH + 'cross_validation/inputs/', fold, train_inputs, val_inputs, test_inputs)
    save_data(DATA_PATH + 'cross_validation/labels/', fold, train_labels, val_labels, test_labels)
    save_data(DATA_PATH + 'cross_validation/input_cameras/', fold, train_input_cameras, val_input_cameras, test_input_cameras)
