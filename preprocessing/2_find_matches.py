import pandas as pd
import numpy as np

DIFFERENCE_THRESHOLD = 0.0015
MAX_TRANSITION_WINDOW = 50  # In frames. Framerate of WNMF is 5FPS.
DATA_PATH = '../data/'


def squared_difference(vector1, vector2):
    return np.mean((vector1 - vector2) * (vector1 - vector2))


def find_possible_entrances(entrances_df, departures_df):
    # Same hour
    possible_entrances = entrances_df[entrances_df['hour'] == departures_df.iloc[departure_index].hour]
    # Different camera
    possible_entrances = possible_entrances[possible_entrances['camera'] != departures_df.iloc[departure_index].camera]
    # Entrance occurs after exit
    possible_entrances = possible_entrances[possible_entrances['frame_num'] > departures_df.iloc[departure_index].frame_num]
    # Entrance occurs within maximum transition time threshold
    possible_entrances = possible_entrances[possible_entrances['frame_num'] < departures_df.iloc[departure_index].frame_num + MAX_TRANSITION_WINDOW]
    return possible_entrances.index


for day_num in range(1, 21):
    day = 'day_' + str(day_num)
    all_tracks = pd.read_csv(DATA_PATH + 'entrances_and_departures/entrances_and_departures_' + day + '.csv')

    num_matches = 0

    # Sort tracks into entrances and departures
    entrances_df = all_tracks[all_tracks['entrance'] == 1]
    departures_df = all_tracks[all_tracks['departure'] == 1]
    departures_df['next_cam'] = 0
    departures_df['next_cam_framenum'] = 0
    entrances_df = entrances_df.reset_index()
    departures_df = departures_df.reset_index()
    entrances_df['entrance_index'] = entrances_df['index']
    departures_df['departure_index'] = departures_df['index']

    # Load pre-computed re-id features
    departure_features = np.load(DATA_PATH + 'reid_features/departure_features_' + day + '.npy')
    entrance_features = np.load(DATA_PATH + 'reid_features/entrance_features_' + day + '.npy')

    print(len(departures_df), len(departure_features))
    print(len(entrances_df), len(entrance_features))

    assert len(departure_features) == len(departures_df)
    assert len(entrance_features) == len(entrances_df)

    # Dataframe to store candidates to be  manually verified
    candidate_matches = pd.DataFrame()
    candidate_matches['difference'] = 0
    candidate_matches['transition_time'] = 100
    candidate_matches['match_num'] = 0
    candidate_matches['entrance_index'] = 0
    candidate_matches['departure_index'] = 0

    for departure_index, departure_feature in enumerate(departure_features):
        print(departure_index, ' of ', len(departure_features))
        candidate_entrance_indexes = find_possible_entrances(entrances_df, departures_df)
        for entrance_index in candidate_entrance_indexes:
            entrance_feature = entrance_features[entrance_index]
            difference = squared_difference(departure_feature, entrance_feature)
            if (difference < DIFFERENCE_THRESHOLD):
                print('match')
                num_matches += 1

                # Match found. Store relevent information in the candidate matches dataframe
                candidate_matches = candidate_matches.append(departures_df.iloc[departure_index])
                candidate_matches.iloc[-1, candidate_matches.columns.get_loc('next_cam')] = entrances_df.iloc[entrance_index].camera
                candidate_matches.iloc[-1, candidate_matches.columns.get_loc('next_cam_framenum')
                                       ] = entrances_df.iloc[entrance_index].frame_num
                candidate_matches.iloc[-1, candidate_matches.columns.get_loc('difference')] = difference
                candidate_matches.iloc[-1, candidate_matches.columns.get_loc(
                    'transition_time')] = entrances_df.iloc[entrance_index].frame_num - departures_df.iloc[departure_index].frame_num
                candidate_matches.iloc[-1, candidate_matches.columns.get_loc('match_num')] = num_matches
                candidate_matches.iloc[-1, candidate_matches.columns.get_loc('entrance_index')] = entrances_df.iloc[entrance_index].entrance_index
                candidate_matches.iloc[-1, candidate_matches.columns.get_loc('departure_index')] = departures_df.iloc[departure_index].departure_index

    # Set all columns but the difference to integers
    candidate_matches[candidate_matches.columns.tolist()[1:]] = candidate_matches[candidate_matches.columns.tolist()[1:]].astype(int)

    candidate_matches.to_csv(DATA_PATH + 'cross_camera_matches/unverified/' + day + '.csv', index=False)
