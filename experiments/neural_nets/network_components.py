import torch.nn as nn


def linear_classifier(input_size, num_hidden, output_size, dropout):
    classifier = nn.Sequential(
        nn.Linear(input_size, num_hidden),
        nn.Dropout(dropout, inplace=True),
        nn.ReLU(inplace=True),
        nn.Linear(num_hidden, output_size),
    )
    return classifier


def linear_decoder(input_size, output_size):
    classifier = nn.Linear(input_size, output_size)
    return classifier


def gru_encoder(input_size, num_hidden):
    encoder = nn.GRUCell(input_size, num_hidden)
    return encoder


def lstm_encoder(input_size, num_hidden):
    encoder = nn.LSTMCell(input_size, num_hidden)
    return encoder
