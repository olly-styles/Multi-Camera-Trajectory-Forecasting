import pandas as pd
import numpy as np

# Experimental setup
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
MIN_BOX_WIDTH = 35
MIN_BOX_HEIGHT = 70
MIN_TRACK_LENGTH = 20
DATA_PATH = '../data/'


def get_track_length(df):
    '''
    Given a dataframe, returns the dataframe with track length field
    df: Dataframe of bounding boxes with columns
    ['filename', 'frame_num', 'track']
    '''
    df['track_length'] = 0
    # Setting to type string to get the order to match the order of videos.
    # This is specific to the WNMF dataset videos.
    df[['hour', 'camera']] = df[['hour', 'camera']].astype(str)
    df = df.sort_values(by=['hour', 'camera', 'track', 'frame_num'])
    df = df.reset_index()
    del df['level_0']
    del df['index']
    all_lengths = np.zeros((len(df)), dtype='uint16')

    # Iterating through rows like this is very innefficient and the bottleneck.
    # This loop should be vectorized for performance.
    for (previous_ix, previous_row), (ix, row) in zip(df.iterrows(), df[1:].iterrows()):
        if ix % 10000 == 0:
            print(ix, ' of ', len(df))
        if continuous_track(row, previous_row):
            all_lengths[ix] = all_lengths[ix - 1] + 1
    df['track_length'] = all_lengths
    return df


def find_entrances_and_departures(df, min_detection_length):
    '''
    Flag bounding_boxes that are labelled. For departure to be labelled, it must have
    a history of a least min_detection_length and be the last frame in the track
    For a entrance to be labeled, it must have a future of at least min_detection_length
    and be the first frame in the track
    '''
    df['max_track_len'] = df.groupby(['hour', 'camera', 'track'])['track_length'].transform(np.max)
    df['departure'] = np.where((df['track_length'] >= min_detection_length) & (
        df['track_length'] == df['max_track_len']), 1, 0)
    df['entrance'] = np.where((df['track_length'] == min_detection_length - 1), 1, 0)
    df['entrance'] = df['entrance'].shift(periods=-(min_detection_length - 1))
    df['entrance'] = df['entrance'].fillna(0).astype(int)
    return df


def continuous_track(row, previous_row):
    return (previous_row['track'] == row['track']) &\
           (previous_row['frame_num'] == row['frame_num'] - 1) &\
           (previous_row['camera'] == row['camera']) &\
           (previous_row['hour'] == row['hour'])


for day_num in range(1, 21):
    day = 'day_' + str(day_num)
    print(day)
    bounding_boxes = pd.read_csv(
        DATA_PATH + 'bounding_boxes/all_bounding_boxes_' + day + '.csv')
    # Clip BB coordinates
    bounding_boxes['x1'] = bounding_boxes['x1'].clip(0, IMAGE_WIDTH)
    bounding_boxes['x2'] = bounding_boxes['x2'].clip(0, IMAGE_WIDTH)
    bounding_boxes['y1'] = bounding_boxes['y1'].clip(0, IMAGE_HEIGHT)
    bounding_boxes['y2'] = bounding_boxes['y2'].clip(0, IMAGE_HEIGHT)

    # Get width and height
    bounding_boxes['w'] = bounding_boxes['x2'] - bounding_boxes['x1']
    bounding_boxes['h'] = bounding_boxes['y2'] - bounding_boxes['y1']

    # Filter width and height
    bounding_boxes = bounding_boxes[bounding_boxes['w'] >= MIN_BOX_WIDTH]
    bounding_boxes = bounding_boxes[bounding_boxes['h'] >= MIN_BOX_HEIGHT]
    bounding_boxes = bounding_boxes[bounding_boxes['h'] > bounding_boxes['w']]

    bounding_boxes = bounding_boxes.sort_values(by=['hour', 'camera', 'track', 'frame_num'])
    bounding_boxes = bounding_boxes.reset_index()

    # Count track length and find entrances and departures
    counted_bounding_boxes = get_track_length(bounding_boxes)
    labeled_bounding_boxes = find_entrances_and_departures(counted_bounding_boxes, MIN_TRACK_LENGTH)

    # Store results
    labeled_bounding_boxes.to_csv(DATA_PATH + 'entrances_and_departures/entrances_and_departures_' + day + '.csv', index=False)
