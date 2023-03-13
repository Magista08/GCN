# PyTorch
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

# Basic
import copy
import os

# Global Variable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Main Code
def train(_model, _data_set, _parameter, log=True):
    model             = _model.to(DEVICE)
    features          = _data_set.features.to(DEVICE)
    train_labels      = _data_set.train_labels.to(DEVICE)
    adjacency_matrix = _data_set.adjacency_matrix.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=_parameter.learning_rate, weight_decay=_parameter.weight_decay)

    # Learning rate
    def learning_rate_lambda(current_step: int):
        if current_step < _parameter.num_warmup_steps:
            return float(current_step) / float(max(1, _parameter.num_warmup_steps))
        return max(0.0, float(_parameter.num_epochs - current_step) /
                   float(max(1, _parameter.num_epochs - _parameter.num_warmup_steps)))

    scheduler = lr_scheduler.LambdaLR(optimizer, learning_rate_lambda)

    if log:
        print("Training started:")
        print(f"\tNum Epochs = {_parameter.num_epochs}")

    least_loss, best_accuracy = float("inf"), 0
    best_model_state_dict     = None
    train_iterator = tqdm(range(0, int(_parameter.num_epochs)), desc="Epoch")

    for epoch in train_iterator:
        model.train()
        outputs = model(features, adjacency_matrix, train_labels)
        loss    = outputs[1]

        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        val_loss, val_accuracy = evaluate(model, features, _data_set.val_labels, adjacency_matrix)
        train_iterator.set_description(f"Training loss = {loss.item():.2f}, "
                                       f"val loss = {val_loss:.2f}, val accuracy = {val_accuracy:.2f}")

        save_best_model = val_loss < least_loss
        if save_best_model:
            least_loss = val_loss
            best_accuracy = val_accuracy
            best_model_state_dict = copy.deepcopy(model.state_dict())
        if save_best_model or _parameter.save_each_epoch or epoch + 1 == _parameter.num_epochs:
            output_dir = os.path.join(_parameter.output_dir, f"Epoch_{epoch + 1}")
            save(model, output_dir)
    if log:
        print(f"Best model val CE loss = {least_loss:.2f}, best model val accuracy = {best_accuracy:.2f}")
        # reloads the best model state dict, bit hacky :P
    model.load_state_dict(best_model_state_dict)


# Evaluate
def evaluate(_model, _features, _test_labels, _additional_matrix):
    features = _features.to(DEVICE)
    test_labels = _test_labels.to(DEVICE)
    additional_matrix = _additional_matrix.to(DEVICE)

    _model.eval()

    outputs = _model(features, additional_matrix, test_labels)
    ce_loss = outputs[1].item()

    ignore_label = nn.CrossEntropyLoss().ignore_index
    predicted_label = torch.max(outputs[0], dim=1).indices[test_labels != ignore_label]
    true_label = test_labels[test_labels != -100]
    accuracy = torch.mean((true_label == predicted_label).type(torch.FloatTensor)).item()

    return ce_loss, accuracy


# Save
def save(_model, _output_dir):
    if not os.path.isdir(_output_dir):
        os.makedirs(_output_dir)

    model_path = os.path.join(_output_dir, "model.pth")
    torch.save(_model.state_dict(), model_path)
