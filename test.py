from data_loader import dataloader

DATA_PATH = "train_dataset/1.npy"
count = 0
train_data = dataloader(DATA_PATH)

for data in train_data:
    print(data)
    count += 1
    if count == 3:
        break