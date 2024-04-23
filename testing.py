import numpy as np
from model import DenseNet
import torch
import torch.nn as nn
from montage_loader import PreprocessedEEGDataset
import torchinfo
from rich import print
import sys

if sys.platform == "darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()


model = DenseNet(
    layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=4, classes=6
)  # model
# torchinfo.summary(model)

LEARNING_RATE = 0.001

loss_fn = nn.KLDivLoss(reduction="batchmean")  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # optimizer
model = model.to(device)


MODEL_PATH = f"./saved_models/data_50only_2.pth"

checkpoint = torch.load(MODEL_PATH)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

# evaluation mode

model.eval()

dataset = PreprocessedEEGDataset("train_montage_cleaned_10k")

# just the first sample as a proof of concept
X, y = dataset[0]

with torch.inference_mode():

    X = torch.from_numpy(X).unsqueeze(0).to(device)
    y = torch.from_numpy(y).to(device)

    test_pred = model(X)
    test_loss = loss_fn(test_pred, y)
    print(f"KL div loss: {test_loss:.5f}")


# multiple samples as a proof of concept
# HACK: the following is just the WORST way to do this
X = np.array([dataset[0][0], dataset[1][0], dataset[2][0]])
y = np.array([dataset[0][1], dataset[1][1], dataset[2][1]])

with torch.inference_mode():

    X = torch.from_numpy(X).to(
        device
    )  # no need to unsqueeze since we have multiple samples
    y = torch.from_numpy(y).to(device)

    test_pred = model(X)
    test_loss = loss_fn(
        test_pred, y
    )  # this will give us the average loss over the input samples

    print(f"KL div loss: {test_loss:.5f}")
