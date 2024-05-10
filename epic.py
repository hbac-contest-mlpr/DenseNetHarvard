from montage_loader import PreprocessedEEGDataset
import numpy as np
import pickle
dataset = PreprocessedEEGDataset("train_montage_cleaned_10k")
indices = pickle.load(open("indices.pkl", "rb"))
dataset[indices]
# freqdict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}  
# indices = []
# i=0
# while True:
#     if freqdict[np.argmax(dataset[i][1])] < 200:
#         freqdict[np.argmax(dataset[i][1])] += 1  
#         indices.append(i)
#     if freqdict[0] == 200 and freqdict[1] == 200 and freqdict[2] == 200 and freqdict[3] == 200 and freqdict[4] == 200 and freqdict[5] == 200:
#         break
#     i+=1
# print(freqdict)
# pickle.dump(indices, open("indices.pkl", "wb"))
# print(indices)