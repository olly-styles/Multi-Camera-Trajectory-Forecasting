import torch
from tqdm import tqdm
import numpy as np


def train(model, device, train_loader, optimizer, loss_function, debug_mode):
    model.train()
    loss = 0
    correct_1 = 0
    correct_3 = 0

    running_total = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if debug_mode and batch_idx > 50:
            break

        # Read data
        inputs = batch['inputs'].to(device)
        input_cameras = batch['input_cameras'].to(device)
        labels = batch['labels'].to(device)
        inputs = inputs.float()
        labels = labels.long()

        assert torch.all(input_cameras != labels)

        # Forward
        outputs = model(inputs, input_cameras)
        loss = loss_function(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss += loss.item()

        # Compute metrics
        for this_out, this_label in zip(outputs, labels):
            _, i = torch.topk(this_out.data, 3)
            if this_label in i[:1]:
                correct_1 += 1
            if this_label in i[:3]:
                correct_3 += 1
        running_total += labels.size(0)

    top_1 = (correct_1 / running_total) * 100
    top_3 = (correct_3 / running_total) * 100
    loss /= len(train_loader)

    return loss, top_1, top_3


def test(model, device, test_loader, loss_function, debug_mode, fold_num=None, predictions_save_path=None):
    model.eval()
    loss = 0
    correct_1 = 0
    correct_3 = 0
    running_total = 0
    all_predictions = np.zeros((0, 15))

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if debug_mode and batch_idx > 50:
                break

            # Read data
            inputs = batch['inputs'].to(device)
            input_cameras = batch['input_cameras'].to(device)
            labels = batch['labels'].to(device)
            inputs = inputs.float()
            labels = labels.long()

            assert torch.all(input_cameras != labels)

            # Forward
            outputs = model(inputs, input_cameras)
            loss = loss_function(outputs, labels)

            if predictions_save_path is not None:
                predictions = np.argsort(-outputs.cpu().numpy()) + 1
                all_predictions = np.append(all_predictions, predictions, axis=0)

            # Compute metrics
            for this_out, this_label in zip(outputs, labels):
                _, i = torch.topk(this_out.data, 3)
                if this_label in i[:1]:
                    correct_1 += 1
                if this_label in i[:3]:
                    correct_3 += 1
            running_total += labels.size(0)

        top_1 = (correct_1 / running_total) * 100
        top_3 = (correct_3 / running_total) * 100
        loss /= len(test_loader)

    if predictions_save_path is not None:
        np.save(predictions_save_path + 'fold_' + str(fold_num) + '.npy', all_predictions)

    return loss, top_1, top_3
