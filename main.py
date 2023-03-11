# Data Collection
import torch.cuda
from utils import data_load

# Model
from models import GCN

# Training
from train import train

# Parameters
LAYER_NUM = 3
INPUT_LOC = "./data/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Parameters:
    learning_rate    = 0.1
    num_epochs       = 200
    weight_decay     = 5e-4
    num_warmup_steps = 0
    save_each_epoch  = False
    output_dir       = "./output/"


if __name__ == '__main__':
    # Basic Data
    input_data = data_load(INPUT_LOC)
    parameters = Parameters

    # init model
    model = GCN(
        _input_size=input_data.features.size(1),
        _hidden_size=16,
        _output_size=input_data.num_classes,
        _dropout=0.5
    )

    # Trainning
    train(model, input_data, parameters, log=True)
