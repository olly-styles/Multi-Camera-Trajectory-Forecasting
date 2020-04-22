from network_components import linear_classifier, linear_decoder, gru_encoder, lstm_encoder
import torch.nn as nn
import torch


class FullyConnectedNetwork(nn.Module):
    def __init__(self, device, num_cameras, input_size, num_hidden_units, output_size, dropout):
        super(FullyConnectedNetwork, self).__init__()
        self.device = device
        self.classifiers = []
        for camera_num in range(num_cameras):
            self.classifiers.append(linear_classifier(input_size, num_hidden_units, output_size, dropout))
        for i, classifier in enumerate(self.classifiers):
            self.add_module(str(i), classifier)

    def forward(self, input, cam):
        outputs = []
        # Loop through batch as different input cameras are processed by differenet classifiers
        for this_input, this_camera in zip(input, cam):
            output = self.classifiers[this_camera](this_input)
            outputs += [output]

        outputs = torch.stack(outputs, 0)
        return outputs


class RecurrentNetwork(nn.Module):
    def __init__(self, device, num_cameras, input_size, num_hidden_units, output_size, recurrence_type):
        super(RecurrentNetwork, self).__init__()
        self.device = device
        self.encoders = []
        self.decoders = []
        self.num_hidden_units = num_hidden_units
        self.recurrence_type = recurrence_type
        if recurrence_type == 'gru':
            for _ in range(num_cameras):
                self.encoders.append(gru_encoder(input_size, num_hidden_units))
                self.decoders.append(linear_decoder(num_hidden_units, output_size))
        elif recurrence_type == 'lstm':
            for _ in range(num_cameras):
                self.encoders.append(lstm_encoder(input_size, num_hidden_units))
                self.decoders.append(linear_decoder(num_hidden_units, output_size))
        else:
            print('Error: Recurrence type ', recurrence_type, ' not recognized')
            exit(1)

        for i, (encoder, decoder) in enumerate(zip(self.encoders, self.decoders)):
            self.add_module(str((i * 2)), encoder)
            self.add_module(str((i * 2) + 1), decoder)

    def forward(self, input, cam):
        outputs = []
        # Loop through batch as different input cameras are processed by differenet classifiers
        for this_input, this_camera in zip(input, cam):
            hidden_unit = torch.zeros(1, self.num_hidden_units, dtype=torch.float).to(self.device)
            # GRU Encoder
            if self.recurrence_type == 'gru':
                for i in range(this_input.size()[0]):
                    inp = this_input[i, :]
                    inp = inp.unsqueeze(0)
                    hidden_unit = self.encoders[this_camera](inp, hidden_unit)
            # LSTM encoder
            elif self.recurrence_type == 'lstm':
                context_unit = torch.zeros(1, self.num_hidden_units, dtype=torch.float).to(self.device)
                for i in range(this_input.size()[0]):
                    inp = this_input[i, :]
                    inp = inp.unsqueeze(0)
                    hidden_unit, context_unit = self.encoders[this_camera](inp, (hidden_unit, context_unit))
            else:
                print('Error: Recurrence type ', self.recurrence_type, ' not recognized')
                exit(1)

            # Decoder
            output = self.decoders[this_camera](hidden_unit)
            outputs += [output]
        outputs = torch.stack(outputs, 0).squeeze()
        return outputs


class LinearClassifier(nn.Module):
    def __init__(self, device, num_cameras, input_size, output_size):
        super(LinearClassifier, self).__init__()
        self.device = device
        self.classifiers = []
        for camera_num in range(num_cameras):
            self.classifiers.append(linear_decoder(input_size, output_size))
        for i, classifier in enumerate(self.classifiers):
            self.add_module(str(i), classifier)

    def forward(self, input, cam):
        outputs = []
        # Loop through batch as different input cameras are processed by differenet classifiers
        for this_input, this_camera in zip(input, cam):
            output = self.classifiers[this_camera](this_input)
            outputs += [output]

        outputs = torch.stack(outputs, 0)
        return outputs
