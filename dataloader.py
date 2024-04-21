import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
BASE_PATH = "../hms-harmful-brain-activity-classification"

class CFG:
    num_classes = 6 # Number of classes in the dataset
    class_names = ['Seizure', 'LPD', 'GPD', 'LRDA','GRDA', 'Other']
    label2name = dict(enumerate(class_names))
    name2label = {v:k for k, v in label2name.items()}
    
sampling_rate =  200

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def load_data(no_of_eeg_id = 100,no_of_subid = 3):
    data = pd.read_csv(f'{BASE_PATH}/train.csv')
    df = data.copy()
    df['eeg_path'] = f'{BASE_PATH}/train_eegs/'+df['eeg_id'].astype(str)+'.parquet'
    df['spec_path'] = f'{BASE_PATH}/train_spectrograms/'+df['spectrogram_id'].astype(str)+'.parquet'
    df['class_name'] = df.expert_consensus.copy()
    df['class_label'] = df.expert_consensus.map(CFG.name2label)

    df_counts = df.groupby('eeg_id')["eeg_id"].count()
    df_filtered = df[df['eeg_id'].isin(df_counts[df_counts >= no_of_subid].index)].copy()
    list_of_eegs = df_filtered.groupby('eeg_id')["eeg_id"].count().index.tolist()
    x =[]
    y = []
    eeg_arr = {}
    class_votes_names = ["gpd_vote","grda_vote","lpd_vote","lrda_vote","other_vote","seizure_vote"]
    df_filtered.loc[:,"softmax"] = df_filtered[class_votes_names].apply(lambda x: softmax(x.tolist()),axis=1) 
    total_eegs_taken = 0
    for eeg_id in tqdm(list_of_eegs[0:no_of_eeg_id]):
        df_multiple = df_filtered[df_filtered['eeg_id']==eeg_id]
        len_df = len(df_multiple)
        parquet_path = f'{BASE_PATH}/train_eegs/{eeg_id}.parquet'
        eeg_data = pd.read_parquet(parquet_path)
        eeg_arr[eeg_id] = eeg_data
        for i in range(0,len_df):
            total_eegs_taken+=1
            eeg_label_offset_seconds=df_multiple[df_multiple["eeg_sub_id"]==i]["eeg_label_offset_seconds"].values[0]
            start_ind_sub_data = int(eeg_label_offset_seconds*sampling_rate)
            end_ind_sub_data = int((eeg_label_offset_seconds + 50) * sampling_rate)
            eeg_sub_data = eeg_data[start_ind_sub_data:end_ind_sub_data]["Fp1"].values
            x.append(eeg_sub_data)
            y.append(df_multiple[df_multiple["eeg_sub_id"]==i]["softmax"].values[0])
            if np.count_nonzero(np.isnan(eeg_sub_data)) > 0:
                inv_nan = ~np.isnan(eeg_sub_data)
                xp = inv_nan.nonzero()[0] # getting the indices of non-nan values
                fp = eeg_sub_data[inv_nan] # getting the values of non-nan values
                n  = np.isnan(eeg_sub_data).nonzero()[0] # getting the indices of nan values
                eeg_sub_data[np.isnan(eeg_sub_data)] = np.interp(n, xp, fp) # linear interpolation
    print(f"Total unique eegs taken: {no_of_eeg_id} , Total eegs taken (counting sub_ids): {total_eegs_taken}")
    X = np.array(x)
    Y = np.array(y)
    return X,Y

