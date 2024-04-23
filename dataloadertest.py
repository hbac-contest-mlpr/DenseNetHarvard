import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
from rich import print

BASE_PATH = Path(
    "/Users/pranjalrastogi/Desktop/SEM4/MLPR/project/data/competition-data"
)


class CFG:
    num_classes = 6  # Number of classes in the dataset
    class_names = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
    label2name = dict(enumerate(class_names))
    name2label = {v: k for k, v in label2name.items()}


sampling_rate = 200


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def montage_from_eeg(eeg):
    montage = np.array([])
    for k in range(4):
        signals = np.zeros(eeg.shape[0])
        COLS = FEATS[k]
        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
            signals += x
        signals = np.array([i for i in signals if not np.isnan(i)])

        if k == 0:
            montage = signals
        else:
            montage = np.vstack((montage, signals))

    return montage


def load_data(no_of_eeg_id=100, no_of_subid=3):
    data = pd.read_csv(f"{BASE_PATH}/train.csv")
    df = data.copy()
    df["eeg_path"] = f"{BASE_PATH}/train_eegs/" + df["eeg_id"].astype(str) + ".parquet"
    df["spec_path"] = (
        f"{BASE_PATH}/train_spectrograms/"
        + df["spectrogram_id"].astype(str)
        + ".parquet"
    )
    df["class_name"] = df.expert_consensus.copy()
    df["class_label"] = df.expert_consensus.map(CFG.name2label)

    df_counts = df.groupby("eeg_id")["eeg_id"].count()
    df_filtered = df[
        df["eeg_id"].isin(df_counts[df_counts >= no_of_subid].index)
    ].copy()
    list_of_eegs = df_filtered.groupby("eeg_id")["eeg_id"].count().index.tolist()
    x = []
    y = []
    class_votes_names = [
        "gpd_vote",
        "grda_vote",
        "lpd_vote",
        "lrda_vote",
        "other_vote",
        "seizure_vote",
    ]
    df_filtered.loc[:, "softmax"] = df_filtered[class_votes_names].apply(
        lambda x: softmax(x.tolist()), axis=1
    )
    total_eegs_taken = 0
    for eeg_id in tqdm(list_of_eegs[0:no_of_eeg_id]):
        df_multiple = df_filtered[df_filtered["eeg_id"] == eeg_id]
        len_df = len(df_multiple)
        parquet_path = f"{BASE_PATH}/train_eegs/{eeg_id}.parquet"
        eeg_data = pd.read_parquet(parquet_path).drop(columns=["EKG"])
        eeg_data = montage_from_eeg(eeg_data)

        # testing montage
        for i in range(0, len_df):
            eeg_label_offset_seconds = df_multiple[df_multiple["eeg_sub_id"] == i][
                "eeg_label_offset_seconds"
            ].values[0]
            start_ind_sub_data = int(eeg_label_offset_seconds * sampling_rate)
            end_ind_sub_data = int((eeg_label_offset_seconds + 50) * sampling_rate)
            eeg_sub_data = eeg_data[:, start_ind_sub_data:end_ind_sub_data]
            if eeg_sub_data.shape[1] == 10000:
                x.append(eeg_sub_data)
                total_eegs_taken += 1
                y.append(
                    df_multiple[df_multiple["eeg_sub_id"] == i]["softmax"].values[0]
                )

    print(
        f"Total unique eegs taken: {no_of_eeg_id} , Total eegs taken (counting sub_ids): {total_eegs_taken}"
    )
    print(len(x))
    X = np.array(x)
    Y = np.array(y)
    return X, Y


def load_cleaned_data(num_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:

    metadata = pd.read_csv(BASE_PATH / "train.csv")
    data_directory = BASE_PATH / "train_montage_cleaned_10k"

    data_paths = list(data_directory.glob("*.npy"))

    Xs = []
    ys = []
    c = 0
    for file_path in tqdm(data_paths, total=num_samples):
        if c == num_samples:
            break

        eeg_id = file_path.name.split("_")[0]
        sub_id = file_path.name.split("_")[1].split(".")[0]

        metadata_row = metadata.query(f"eeg_id == {eeg_id}")
        row_info = metadata_row.loc[
            int(sub_id),
            [
                "expert_consensus",
                "gpd_vote",
                "lpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
                "seizure_vote",
            ],
        ]
        x = row_info[["gpd_vote","grda_vote","lpd_vote","lrda_vote","other_vote","seizure_vote"]].tolist()
        row_info.loc[["gpd_vote","grda_vote","lpd_vote","lrda_vote","other_vote","seizure_vote"]] = softmax(x)
        
        X = np.load(file_path)
        y = row_info[["gpd_vote","grda_vote","lpd_vote","lrda_vote","other_vote","seizure_vote"]].values
        y = y.astype(np.float32)

        Xs.append(X)
        ys.append(y)

        c += 1

    Xs = np.array(Xs)
    ys = np.array(ys)

    print(f"Loaded {c} samples | X shape: {Xs.shape} | y shape: {ys.shape}")
    return Xs, ys


if __name__ == "__main__":
    load_cleaned_data()
