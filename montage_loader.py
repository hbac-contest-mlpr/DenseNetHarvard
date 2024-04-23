from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

BASE_PATH = Path(
    "/Users/pranjalrastogi/Desktop/SEM4/MLPR/project/data/competition-data"
)

class PreprocessedEEGDataset(Dataset):
    def __init__(self, image_data_directory):
        data_directory = BASE_PATH / image_data_directory
        
        self.data_paths = list(data_directory.glob("*.npy"))
        self.length = len(self.data_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        metadata = pd.read_csv(BASE_PATH / "train.csv")
        data_path = self.data_paths[idx]

        eeg_id = data_path.name.split("_")[0]
        sub_id = data_path.name.split("_")[1].split(".")[0]

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
        
        X = np.load(data_path)
        X = X.astype(np.float32)
        y = row_info[["gpd_vote","grda_vote","lpd_vote","lrda_vote","other_vote","seizure_vote"]].values
        y = y.astype(np.float32)

        return X, y

