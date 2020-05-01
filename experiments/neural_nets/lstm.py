# Internal modules
import models
import datasets
import trainer
# External modules
import torch
import torch.optim as optim
import torch.nn as nn
import copy

# ########## NETWORK CONFIG ########## #
DEVICE = torch.device('cuda')
INPUT_SIZE = 4
NUM_HIDDEN_UNITS = 128
OUTPUT_SIZE = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_WORKERS = 4
WEIGHT_DECAY = 0
NUM_EPOCHS = 10
DEBUG_MODE = False

# ########## DATASET CONFIG ########## #
NUM_CAMERAS = 15
DATA_PATH = '../../data/'
MODEL_SAVE_PATH = DATA_PATH + 'models/lstm/'
PREDICTIONS_SAVE_PATH = DATA_PATH + 'predictions/lstm/'
INPUT_DATA_PATH = DATA_PATH + 'cross_validation/'


network_args = {'device': DEVICE,
                'num_cameras': NUM_CAMERAS,
                'input_size': INPUT_SIZE,
                'num_hidden_units': NUM_HIDDEN_UNITS,
                'output_size': OUTPUT_SIZE,
                'recurrence_type': 'lstm'}


for fold in range(1, 6):

    # ########## SET UP DATASET ########## #
    train_dataset_args = {'data_path': INPUT_DATA_PATH,
                          'fold': fold,
                          'phase': 'train',
                          'batch_size': BATCH_SIZE,
                          'shuffle': True,
                          'num_workers': NUM_WORKERS,
                          'flatten_inputs': False}

    val_dataset_args = {'data_path': INPUT_DATA_PATH,
                        'fold': fold,
                        'phase': 'val',
                        'batch_size': BATCH_SIZE,
                        'shuffle': False,
                        'num_workers': NUM_WORKERS,
                        'flatten_inputs': False}

    test_dataset_args = {'data_path': INPUT_DATA_PATH,
                         'fold': fold,
                         'phase': 'test',
                         'batch_size': BATCH_SIZE,
                         'shuffle': False,
                         'num_workers': NUM_WORKERS,
                         'flatten_inputs': False}

    train_loader = datasets.get_classification_dataset(**train_dataset_args)
    val_loader = datasets.get_classification_dataset(**val_dataset_args)
    test_loader = datasets.get_classification_dataset(**test_dataset_args)

    # ########## SET UP MODEL ########## #
    model = models.RecurrentNetwork(**network_args).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.CrossEntropyLoss()

    # ########## TRAIN AND EVALUATE ########## #
    best_score = 0

    for epoch in range(NUM_EPOCHS):
        print('----------- EPOCH ' + str(epoch) + ' -----------')

        trainer_args = {'model': model,
                        'device': DEVICE,
                        'train_loader': train_loader,
                        'optimizer': optimizer,
                        'loss_function': loss_function,
                        'debug_mode': DEBUG_MODE}

        loss, top_1, top_3 = trainer.train(**trainer_args)
        print('Train loss: {0:.5f} Top 1: {1:.1f}% Top 3: {2:.1f}%'.format(loss, top_1, top_3))

        val_args = {'model': model,
                    'device': DEVICE,
                    'test_loader': val_loader,
                    'loss_function': loss_function,
                    'debug_mode': DEBUG_MODE}

        loss, top_1, top_3 = trainer.test(**val_args)
        print('Validation loss: {0:.5f} Top 1: {1:.1f}% Top 3: {2:.1f}%'.format(loss, top_1, top_3))

        if top_3 > best_score:
            best_top_3 = top_3
            best_model = copy.deepcopy(model)
            model_save_name = 'fold_' + str(fold) + '.weights'
            torch.save(best_model.state_dict(), MODEL_SAVE_PATH + model_save_name)

    test_args = {'model': model,
                 'device': DEVICE,
                 'test_loader': test_loader,
                 'loss_function': loss_function,
                 'debug_mode': False,
                 'fold_num': fold,
                 'predictions_save_path': PREDICTIONS_SAVE_PATH}

    loss, top_1, top_3 = trainer.test(**test_args)
    print('Test loss: {0:.5f} Top 1: {1:.1f}% Top 3: {2:.1f}%'.format(loss, top_1, top_3))
