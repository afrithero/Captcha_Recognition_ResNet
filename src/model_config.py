import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_FREQ = 10
EPOCHS = 1
BATCH_SIZE = 32
LR = 0.0003