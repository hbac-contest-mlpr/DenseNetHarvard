import numpy as np
from model import DenseNet
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
from montage_loader import PreprocessedEEGDataset
import torchinfo
from rich import print
import sys
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

if sys.platform == "darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

def load_model(model_path="model.pth", print_summary=False):

    model = DenseNet(
        layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=4, classes=6
    )  # model
    if print_summary:
        torchinfo.summary(model)

    loss_fn = nn.KLDivLoss(reduction="batchmean")  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # optimizer
    model = model.to(device)


    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # evaluation mode
    model.eval()

    return model, loss_fn


def sample_tests(model, loss_fn):
    dataset = PreprocessedEEGDataset("train_montage_cleaned_10k")

    # just the first sample as a proof of concept
    X, y = dataset[0]

    with torch.inference_mode():

        X = torch.from_numpy(X).unsqueeze(0).to(device)
        y = torch.from_numpy(y).to(device)

        test_pred = model(X)
        test_loss_singular = loss_fn(test_pred, y)
        print(f"KL div loss: {test_loss_singular:.5f}")


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
        test_loss_batch = loss_fn(
            test_pred, y
        )  # this will give us the average loss over the input samples

    return test_loss_singular, test_loss_batch

def complete_tests(model, loss_fn, batch_size=32, test_size_used=0.3, test_on_train=False, max_batches=None):
    dataset = PreprocessedEEGDataset("train_montage_cleaned_10k")

    torch.manual_seed(0)
    _, test_dataset = random_split(dataset, [1 - test_size_used, test_size_used])
    if test_on_train:
        raise NotImplementedError("Test on train not implemented yet")
    
    # only test on test dataset (since seed is fixed, this will be consistent across runs)
    data_loader = DataLoader(
        test_dataset, batch_size=batch_size
    )

    test_loss = 0

    classes_true = []
    classes_pred = []

    if max_batches is None:
        max_batches = len(data_loader)
    else:
        max_batches = min(max_batches, len(data_loader))
    
    with torch.inference_mode():
        t0 = time.time()
        for batch_idx, (X, y) in enumerate(tqdm(data_loader, total=max_batches, desc="Testing")):

            X = X.to(device)
            y = y.to(device)

            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            classes_pred.extend(np.argmax(test_pred.cpu().detach().numpy(), axis=1))
            classes_true.extend(np.argmax(y.cpu().detach().numpy(), axis=1))

            if batch_idx % 100 == 0:
                tqdm.write(f"Batch {batch_idx}: set loss: {test_loss:.5f} | time elapsed: {time.time() - t0:.2f}s")
            
            if batch_idx >= max_batches:
                break

        test_loss /= max_batches

    return test_loss, np.array(classes_true), np.array(classes_pred)

LEARNING_RATE = 0.001
MODEL_PATH = f"./saved_models/all_data_10.pth"

if __name__ == "__main__":
    model, loss_fn = load_model(MODEL_PATH)

    # test_losses = sample_tests(model, loss_fn)
    # print(f"Test loss for singular sample: {test_losses[0]:.5f}")
    # print(f"Test loss for batch of samples: {test_losses[1]:.5f}")

    classification_result = complete_tests(model, loss_fn, max_batches=200)
    print(f"Test loss for complete dataset: {classification_result[0]:.5f}")
    
    print(classification_report(classification_result[1], classification_result[2]))
    cm = confusion_matrix(classification_result[1], classification_result[2])
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True)
    plt.show()
