from model import DenseNet
import torch
import torch.nn as nn
import time
from montage_loader import PreprocessedEEGDataset
from torch.utils.data import DataLoader, random_split, Subset
from rich import print
import sys
import json
import numpy as np

if sys.platform == "darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

# some hyperparameters
TEST_SIZE = 0.3
LEARNING_RATE = 0.001

# epochs stuff
MAX_EPOCHS = 6
SAVE_EVERY = 1

# memory stuff
BATCH_SIZE = 32

# model stuff
MODEL_PREFIX = "all_data_new2_"  # please add _ at the end
PRESAVED_MODEL_PATH = "./saved_models/all_data_new_5.pth"
USE_SUBSET = False
LEN_SUBSET = 10 # number of samples to use if USE_SUBSET is True
# BATCH_COUNT = LEN_SUBSET // BATCH_SIZE if USE_SUBSET else len(train_dataset) // BATCH_SIZE
# so, set PRINT_EVERY_BATCH properly!
PRINT_EVERY_BATCH = 128

NUM_WORKERS = 4

STATS_SAVE_PATH = "./stats/"  # trailing slash is important!

torch.manual_seed(0)


def print_params():
    print(f"{' Hyperparameters ':=^80}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print(f"Save Every: {SAVE_EVERY}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Model Prefix: '{MODEL_PREFIX}'")
    print(f"Presaved Model Path: '{PRESAVED_MODEL_PATH}'")
    print(f"Use Subset: {USE_SUBSET} (Length of Subset: {LEN_SUBSET})")
    print(f"Print Every Batch: {PRINT_EVERY_BATCH}")
    print(f"Number of Workers: {NUM_WORKERS}")
    print(f"Device: {device}")
    print(f"Stats Save Path: '{STATS_SAVE_PATH}'")
    print(f"{'':=^80}")

def main():
    dataset = PreprocessedEEGDataset("train_montage_cleaned_10k")  # dataset object


    if USE_SUBSET:
        print(f"Using subset of {LEN_SUBSET} samples only! Picked the first {LEN_SUBSET} samples.")
        subset = Subset(dataset, range(0, LEN_SUBSET))
        train_subset, test_subset = random_split(subset, [1 - TEST_SIZE, TEST_SIZE])

        train_batches = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        test_batches = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    else:
        train_dataset, test_dataset = random_split(dataset, [1 - TEST_SIZE, TEST_SIZE])

        train_batches = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        test_batches = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


    model = DenseNet(
        layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=4, classes=6
    )  # model

    loss_fn = nn.KLDivLoss(reduction="batchmean")  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # optimizer

    model = model.to(device)

    if PRESAVED_MODEL_PATH:
        print(f"Trying to load model from '{PRESAVED_MODEL_PATH}'")
        checkpoint = torch.load(PRESAVED_MODEL_PATH)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


    print(f"Length of train_batches: {len(train_batches)}")
    print(f"Length of test_batches: {len(test_batches)}")

    for epoch in range(MAX_EPOCHS):
        print(f"Epoch: {epoch}")
        epoch_statistics = {"epoch": epoch, "test_size": len(test_batches), "train_size": len(train_batches)}
        model.train()

        t0 = bt = time.time()
        overall_loss = 0
        train_classes = []
        pred_classes = []
        for batch_idx, (x_train, y_train) in enumerate(train_batches):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
        
            y_pred = model(x_train)  # forward pass

            train_classes.extend(list(np.argmax(y_pred.cpu().detach().numpy(), axis=1)))
            pred_classes.extend(list(np.argmax(y_train.cpu().detach().numpy(), axis=1)))

            loss = loss_fn(y_pred, y_train)  # calculate loss
            overall_loss += loss
            optimizer.zero_grad()  # zero the gradients

            loss.backward()  # backward pass

            optimizer.step()  # update weights

            if batch_idx % PRINT_EVERY_BATCH == 0:
                print(f"\t Batch_idx: {batch_idx} | Batch Loss: {loss:.5f} (time: {time.time()-bt:.2f}s | {time.time()-t0:.2f}s elapsed)")
            bt = time.time()
        overall_loss /= len(train_batches)

        epoch_statistics["train_loss"] = overall_loss.cpu().detach().numpy().tolist()
        epoch_statistics["train_time"] = time.time() - t0
        epoch_statistics["train_true_classes"] = [int(x) for x in train_classes]
        epoch_statistics["train_pred_classes"] = [int(x) for x in pred_classes]
        print(f"Overall Train Loss: {overall_loss:.5f}")

        with open(f"{STATS_SAVE_PATH}stats_{MODEL_PREFIX}{epoch}.json", "w") as f:
            json.dump(epoch_statistics, f)
        
        del train_classes
        del pred_classes

        epoch_statistics = {}
        with open(f"{STATS_SAVE_PATH}stats_{MODEL_PREFIX}{epoch}.json", "r") as f:
            epoch_statistics = json.load(f)


        model.eval()  # set model to evaluation mode
        tt0 = time.time()
        test_classes = []
        pred_classes = []
        with torch.inference_mode():
            test_loss = 0
            for batch_idx, (x_test, y_test) in enumerate(test_batches):
                x_test = x_test.to(device)
                y_test = y_test.to(device)

                test_pred = model(x_test)  # forward pass
                test_classes.extend(list(np.argmax(test_pred.cpu().detach().numpy(), axis=1)))
                pred_classes.extend(list(np.argmax(y_test.cpu().detach().numpy(), axis=1)))
                test_loss += loss_fn(test_pred, y_test)  # calculate loss
            test_loss /= len(test_batches)

        epoch_statistics["test_loss"] = test_loss.cpu().detach().numpy().tolist()
        epoch_statistics["test_time"] = time.time() - tt0
        epoch_statistics["test_true_classes"] = [int(x) for x in test_classes]
        epoch_statistics["test_pred_classes"] = [int(x) for x in pred_classes]
        print(f"Overall Test loss: {test_loss:.5f}")


        # saving checkpoint
        if epoch % SAVE_EVERY == 0:
            epoch_statistics["saved"] = True
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
        else:
            epoch_statistics["saved"] = False

        t1 = time.time()
        epoch_statistics["epoch_time"] = t1 - t0
        with open(f"{STATS_SAVE_PATH}stats_{MODEL_PREFIX}{epoch}.json", "w") as f:
            json.dump(epoch_statistics, f)
        
        print(f"Time taken for entire epoch: {t1-t0:.2f}s\n")
        
        del test_classes
        del pred_classes

    print(f"{' Training complete ':=^80}")


if __name__ == "__main__":
    print_params()
    main()