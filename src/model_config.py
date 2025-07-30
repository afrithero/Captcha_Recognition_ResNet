import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Linux
SAVE_FREQ = 10
EPOCHS = 100
BATCH_SIZE = 100
LR = 0.0003