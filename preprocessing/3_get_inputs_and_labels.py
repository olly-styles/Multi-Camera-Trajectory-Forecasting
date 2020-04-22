import pandas as pd
import numpy as np

OFFSET_LEN = 10
DEBUG = True
TRAJECTORY_LENGTH = 10
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
DATA_PATH = '../data/'
LABELED_TRACK_PATH = DATA_PATH + 'cross_camera_matches/verified_no_duplicates/'
ENTRANCES_DEPARTURES_PATH = DATA_PATH + 'entrances_and_departures/'

for day_num in range(1, 21):
    labeled_bounding_boxes = pd.read_csv(LABELED_TRACK_PATH + '/day_' + str(day_num) + '.csv')
    all_bounding_boxes = pd.read_csv(ENTRANCES_DEPARTURES_PATH + 'entrances_and_departures_day_' + str(day_num) + '.csv')

    inputs = np.zeros((len(labeled_bounding_boxes) * OFFSET_LEN, TRAJECTORY_LENGTH, 4), dtype='float')
    labels = np.zeros((len(labeled_bounding_boxes) * OFFSET_LEN), dtype='int')
    input_cameras = np.zeros((len(labeled_bounding_boxes) * OFFSET_LEN), dtype='int')
    labeled_bounding_boxes = labeled_bounding_boxes.reset_index()
    for ix, row in labeled_bounding_boxes.iterrows():
        print('Day', day_num, ix, ' of ', len(labeled_bounding_boxes))
        reference_track = row['track']
        for offset in range(0, OFFSET_LEN):
            labels[(ix * OFFSET_LEN) + offset] = row['next_cam'].astype(int)
            input_cameras[(ix * OFFSET_LEN) + offset] = row['camera'].astype(int)
            for t in range(0, TRAJECTORY_LENGTH):
                this_input_track = int(all_bounding_boxes.iloc[int(row['departure_index'] - OFFSET_LEN - offset + t)]['track'])
                assert reference_track == this_input_track
                inputs[(ix * OFFSET_LEN) + offset, t, 0] = int(all_bounding_boxes.iloc[int(row['departure_index'] - OFFSET_LEN - offset + t)]['x1']) / float(IMAGE_WIDTH)
                inputs[(ix * OFFSET_LEN) + offset, t, 1] = int(all_bounding_boxes.iloc[int(row['departure_index'] - OFFSET_LEN - offset + t)]['y1']) / float(IMAGE_HEIGHT)
                inputs[(ix * OFFSET_LEN) + offset, t, 2] = int(all_bounding_boxes.iloc[int(row['departure_index'] - OFFSET_LEN - offset + t)]['x2']) / float(IMAGE_WIDTH)
                inputs[(ix * OFFSET_LEN) + offset, t, 3] = int(all_bounding_boxes.iloc[int(row['departure_index'] - OFFSET_LEN - offset + t)]['y2']) / float(IMAGE_HEIGHT)

    # Next camera is never the current camera
    assert np.all(input_cameras != labels)
    np.save(DATA_PATH + 'inputs/inputs_day' + str(day_num) + '.npy', inputs)
    np.save(DATA_PATH + 'labels/one_class_classification/one_class_classification_label_day' + str(day_num) + '.npy', labels)
    np.save(DATA_PATH + 'input_cameras/input_cameras_day' + str(day_num) + '.npy', input_cameras)
