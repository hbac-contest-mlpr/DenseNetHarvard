from model import DenseNet
import torch
import torch.nn as nn
import time
from montage_loader import PreprocessedEEGDataset
from torch.utils.data import DataLoader, random_split, Subset
from rich import print
import sys

if sys.platform == "darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

# some hyperparameters
TEST_SIZE = 0.3
LEARNING_RATE = 0.001

# epochs stuff
MAX_EPOCHS = 10
SAVE_EVERY = 2

# memory stuff
BATCH_SIZE = 32

# model stuff
MODEL_PREFIX = "data_50only_"  # please add _ at the end
USE_SUBSET = True
LEN_SUBSET = 100
# BATCH_COUNT = LEN_SUBSET // BATCH_SIZE if USE_SUBSET else len(train_dataset) // BATCH_SIZE
# so, set PRINT_EVERY_BATCH properly!
PRINT_EVERY_BATCH = 1

torch.manual_seed(0)

dataset = PreprocessedEEGDataset("train_montage_cleaned_10k")  # dataset object


if USE_SUBSET:
    print(f"Using subset of {LEN_SUBSET} samples only! Picked the first {LEN_SUBSET} samples.")
    subset = Subset(dataset, range(0, LEN_SUBSET))
    train_subset, test_subset = random_split(subset, [1 - TEST_SIZE, TEST_SIZE])

    train_batches = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_batches = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True)
else:
    train_dataset, test_dataset = random_split(dataset, [1 - TEST_SIZE, TEST_SIZE])

    train_batches = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_batches = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = DenseNet(
    layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=4, classes=6
)  # model

loss_fn = nn.KLDivLoss(reduction="batchmean")  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # optimizer

model = model.to(device)


for epoch in range(MAX_EPOCHS):
    print(f"Epoch: {epoch}")

    model.train()

    t0 = time.time()

    for batch_idx, (x_train, y_train) in enumerate(train_batches):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
    
        y_pred = model(x_train)  # forward pass
        loss = loss_fn(y_pred, y_train)  # calculate loss
        optimizer.zero_grad()  # zero the gradients

        loss.backward()  # backward pass

        optimizer.step()  # update weights

        if batch_idx % PRINT_EVERY_BATCH == 0:
            print(f"\t Batch_idx: {batch_idx} | Loss: {loss:.5f}")

    model.eval()  # set model to evaluation mode
    with torch.inference_mode():
        test_loss = 0
        for batch_idx, (x_test, y_test) in enumerate(test_batches):
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            test_pred = model(x_test)  # forward pass
            test_loss += loss_fn(test_pred, y_test)  # calculate loss
        test_loss /= len(test_batches)

    print(f"Test loss: {test_loss:.5f}")

    # saving checkpoint
    if epoch and epoch % SAVE_EVERY == 0:
        print(f"[green]Saving model at epoch {epoch}![/green]")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": test_loss,
            },
            f"./saved_models/{MODEL_PREFIX}{epoch}.pth",
        )

    t1 = time.time()
    print(f"Time taken for entire epoch: {t1-t0:.2f}s\n")

print(f"{' Training complete ':=^80}")
