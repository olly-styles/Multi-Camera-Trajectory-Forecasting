from torch.utils.data import Dataset
import numpy as np
import torch


class ClassificationDataset(Dataset):
    def __init__(self, inputs_path, input_cameras_path, labels_path, flatten_inputs):
        self.inputs = np.load(inputs_path)
        self.input_cameras = np.load(input_cameras_path)
        self.labels = np.load(labels_path)
        self.flatten_inputs = flatten_inputs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_inputs = self.inputs[idx]
        batch_input_cameras = self.input_cameras[idx]
        batch_labels = self.labels[idx]

        # Index from 0
        batch_labels -= 1
        batch_input_cameras -= 1

        if self.flatten_inputs:
            batch_inputs = batch_inputs.flatten()

        return {'inputs': batch_inputs,
                'input_cameras': batch_input_cameras,
                'labels': batch_labels}


def get_classification_dataset(data_path, fold, phase, batch_size, shuffle, num_workers, flatten_inputs):
    filename = phase + '_fold' + str(fold) + '.npy'
    inputs_path = data_path + 'inputs/' + filename
    input_cameras_path = data_path + 'input_cameras/' + filename
    labels_path = data_path + 'labels/' + filename

    dataset = ClassificationDataset(inputs_path, input_cameras_path, labels_path, flatten_inputs)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers)
    return dataset_loader


def get_hand_crafted_classification_dataset(data_path, fold, phase, batch_size, shuffle, num_workers, flatten_inputs):
    filename = phase + '_fold' + str(fold) + '.npy'
    inputs_path = data_path + 'hand_crafted_features/' + filename
    input_cameras_path = data_path + 'input_cameras/' + filename
    labels_path = data_path + 'labels/' + filename

    dataset = ClassificationDataset(inputs_path, input_cameras_path, labels_path, flatten_inputs)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers)
    return dataset_loader
