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

        if isinstance(idx, slice):
            raise NotImplementedError("Slicing is not supported")
        
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
    
class PreprocessedDatasetSingular(Dataset):
    def __init__(self, image_data_directory):
        data_directory = BASE_PATH / image_data_directory
        
        self.data_paths = list(data_directory.glob("*.npy"))

        # group data paths by eeg_id
        self.data_paths = {int(x.name.split("_")[0]): x for x in self.data_paths}
        # because its a dict, it auto chooses one
        self.data_paths_array = list(self.data_paths.keys())
        self.length = len(self.data_paths_array)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            raise NotImplementedError("Slicing is not supported")
        
        metadata = pd.read_csv(BASE_PATH / "train.csv")

        data_path = self.data_paths[self.data_paths_array[idx]]

        eeg_id = data_path.name.split("_")[0]
        sub_id = data_path.name.split("_")[1].split(".")[0]

        metadata_row = metadata.query(f"eeg_id == {eeg_id}")

        label_cols = metadata.columns[-6:]

        aux = metadata_row.groupby('eeg_id')[label_cols].agg('sum')
        softmaxready = aux.iloc[0][["gpd_vote","grda_vote","lpd_vote","lrda_vote","other_vote","seizure_vote"]].to_numpy()

        X = np.load(data_path)
        X = X.astype(np.float32)
        y = softmax(softmaxready)
        y = y.astype(np.float32)

        return X, y
    


class PreprocessedDatasetSingularMiddle10(Dataset):
    def __init__(self, image_data_directory):
        data_directory = BASE_PATH / image_data_directory
        
        self.data_paths = list(data_directory.glob("*.npy"))

        # group data paths by eeg_id
        self.data_paths = {int(x.name.split("_")[0]): x for x in self.data_paths}
        # because its a dict, it auto chooses one
        self.data_paths_array = list(self.data_paths.keys())
        self.length = len(self.data_paths_array)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            raise NotImplementedError("Slicing is not supported")
        
        metadata = pd.read_csv(BASE_PATH / "train.csv")

        data_path = self.data_paths[self.data_paths_array[idx]]

        eeg_id = data_path.name.split("_")[0]
        sub_id = data_path.name.split("_")[1].split(".")[0]

        metadata_row = metadata.query(f"eeg_id == {eeg_id}")

        label_cols = metadata.columns[-6:]

        aux = metadata_row.groupby('eeg_id')[label_cols].agg('sum')
        softmaxready = aux.iloc[0][["gpd_vote","grda_vote","lpd_vote","lrda_vote","other_vote","seizure_vote"]].to_numpy()

        X = np.load(data_path)
        X = X.astype(np.float32)
        X = X[:, 4000:6000]

        y = softmax(softmaxready)
        y = y.astype(np.float32)

        return X, y


if __name__ == "__main__":
    data = PreprocessedDatasetSingularMiddle10("train_montage_cleaned_10k")
    print(len(data))

    x, y = data[0]
    print(y)
    print(x.shape, y.shape)
