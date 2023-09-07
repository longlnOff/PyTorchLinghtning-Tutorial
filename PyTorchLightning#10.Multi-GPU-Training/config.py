import torch


# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 128


# Dataset
DATA_DIR = "dataset"
NUM_WORKERS = 4


# Compute related
ACCERLATOR = "gpu"
DEVICES = [0]
PRECISION = "16-mixed"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
