import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from torchsummary import summary
# from torchvision.transforms import ToTensor
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
from model import *
from sklearn.preprocessing import StandardScaler
from augmentations import *
import logging
from datetime import datetime

device = torch.device(0 if torch.cuda.is_available() else "cpu")
print("Using " + str(device))

# Locate the data
mp_data_path = '/home/prashtata/gradschool/asl/dataset/MP_data'
label_list = os.listdir(mp_data_path) #Since the labels are the directory names, we shall use them

# print(len(label_list))

data = []
labels = []
num_labels = len(label_list)
# num_labels = 300

# Config the logger
now = datetime.now()
logging.basicConfig(
    filename=f'./logs/training_{num_labels}_{now}.log',  # Optional: log to a file
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s' # Format of log messages
)


for i in range(num_labels):

    # List the samples under the label
    label_dir = os.path.join(mp_data_path, label_list[i])
    sample_list = os.listdir(label_dir)

    # Access the samples
    for sample in sample_list:
        kp_file = os.path.join(label_dir, sample, sample+'_keypoints.npy')
        if not os.path.exists(kp_file):
            continue

        keypoints = np.load(kp_file)
        # print(keypoints.shape)

        # Append the keypoints to the data list and the corresponding label to the labels list
        data.append(keypoints); labels.append(i) # Labels will be denoted their index number
        # Add the keypoint augmentations alongside
        data.append(jitter_keypoints(keypoints)); labels.append(i)
        data.append(scale_keypoints(keypoints)); labels.append(i)
        data.append(time_warp_keypoints(keypoints)); labels.append(i)
        data.append(jitter_keypoints(keypoints, noise_level=0.1)); labels.append(i)
        data.append(scale_keypoints(keypoints, scale_range = (0.6, 1.4))); labels.append(i)
        data.append(time_warp_keypoints(keypoints, sigma = 0.4)); labels.append(i)
    
    print(f"Processed files from label# {i+1}/{num_labels}", end='\r')
        

#Standardize the data
scaler = StandardScaler()
data = [scaler.fit_transform(d) for d in data]  # Normalize keypoints
print("Standardization Complete")

# Create a dataset class for the above data
class MP_KeypointDataset(Dataset):
    def __init__(self, data, labels, max_seq_len=30):  # Choose a suitable max_seq_len
        self.max_seq_len = max_seq_len
        self.data = [self.pad_or_truncate(torch.tensor(d, dtype=torch.float32)) for d in data]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def pad_or_truncate(self, tensor):
        """Pads or truncates the input tensor to max_seq_len."""
        seq_len, feature_dim = tensor.shape
        if seq_len > self.max_seq_len:
            return tensor[int((seq_len-self.max_seq_len)/2):int((seq_len+self.max_seq_len)/2)]  # Truncate
        else:
            padding = torch.zeros(self.max_seq_len - seq_len, feature_dim)  # Pad
            return torch.cat((tensor, padding), dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
# Initialize the dataset
keypoint_dataset = MP_KeypointDataset(data, labels)
print("Dataset Object Constructed")

# Split the dataset into train, val and test sets
split_ratio = np.array([0.7, 0.3, 0])
train_dataset, val_dataset, test_dataset = random_split(keypoint_dataset, split_ratio)

train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=6, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=6, shuffle=False)
print("Dataloaders Initialized")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Now to establish model and training parameters parameters
hidden_size = 128
num_stacked_layers = 4

model = LSTM(input_size = len(data[0][0]), output_size = num_labels, hidden_size = hidden_size, num_stacked_layers = num_stacked_layers)
# model = LSTM_divided(input_size = len(data[0][0]), output_size = num_labels, hidden_size = hidden_size)
logging.info(model)
# Send model to GPU
model.to(device)
print(f"Trainable Parameters: {count_parameters(model)}")


learning_rate = 1e-4
epochs = 500

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4)

loss_fn = nn.CrossEntropyLoss()

# Initialize training and val loggers, as well as log
training_loss_log = []
val_loss_log = []

training_acc_log = []
val_acc_log = []


#Begin training
pbar = trange(0, epochs, leave = False, desc="Epoch")
train_acc = 0
val_acc = 0

logging.info("Training started")
for epoch in pbar:
    pbar.set_postfix_str("Accuracy: Train - %.2f, Val - %.2f" % (train_acc*100, val_acc*100))

    model.train()
    steps = 0
    for input, label in tqdm(train_dataloader, desc="Training", leave=False):
        bs = label.shape[0]
        input = input.to(device)
        label = label.to(device)

        hidden1 = torch.zeros(num_stacked_layers, bs, hidden_size, device=device)
        memory1 = torch.zeros(num_stacked_layers, bs, hidden_size, device=device)
        hidden2 = torch.zeros(4, bs, hidden_size, device=device)
        memory2 = torch.zeros(4, bs, hidden_size, device=device)
        pred, hidden1, memory1 = model(input, hidden1, memory1)
        # pred, hidden1, memory1, _, _ = model(input, hidden1, memory1, hidden2, memory2)

        loss = loss_fn(pred[:,-1, :], label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss_log.append(loss.item())
        

        train_acc += (pred[:,-1,:].argmax(1) == label).sum()
        steps+=bs

    
    train_acc = (train_acc/steps).item()
    training_acc_log.append(train_acc)

    model.eval()
    steps = 0
    with torch.no_grad():
        for input, label in tqdm(val_dataloader, desc="Validation", leave=False):
            bs = label.shape[0]
            input = input.to(device)
            label = label.to(device)
        
            hidden1 = torch.zeros(num_stacked_layers, bs, hidden_size, device=device)
            memory1 = torch.zeros(num_stacked_layers, bs, hidden_size, device=device)
            hidden2 = torch.zeros(4, bs, hidden_size, device=device)
            memory2 = torch.zeros(4, bs, hidden_size, device=device)
            pred, hidden1, memory1 = model(input, hidden1, memory1)
            # pred, hidden1, memory1, _, _ = model(input, hidden1, memory1, hidden2, memory2)

            loss = loss_fn(pred[:,-1, :], label)
            val_loss_log.append(loss.item())

            val_acc += (pred[:,-1,:].argmax(1) == label).sum()
            steps+=bs
        
        val_acc = (val_acc/steps).item()
        val_acc_log.append(val_acc)

    # scheduler.step(loss)

    logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Training Accuracy: {train_acc: .4f}, Validation Accuracy: {val_acc: 0.4f}")

logging.info("Training finished")


# Plot the training and validation losses

_ = plt.figure(figsize=(10,5))
_ = plt.plot(np.linspace(0, epochs, len(training_loss_log)), training_loss_log)
_ = plt.plot(np.linspace(0, epochs, len(val_loss_log)), val_loss_log)
_ = plt.legend(["Training", "Validation"])
_ = plt.title("Train vs. Val Loss")
_ = plt.xlabel("Epochs")
_ = plt.ylabel("Loss")
_ = plt.show()
_ = plt.savefig(f"./logs/loss_{num_labels}_{now}.png")

_ = plt.figure(figsize=(10,5))
_ = plt.plot(np.linspace(0, epochs, len(training_acc_log)), training_acc_log)
_ = plt.plot(np.linspace(0, epochs, len(val_acc_log)), val_acc_log)
_ = plt.legend(["Training", "Validation"])
_ = plt.title("Train vs. Val Accuracy")
_ = plt.xlabel("Epochs")
_ = plt.ylabel("Accuracy")
_ = plt.show()
_ = plt.savefig(f"./logs/accuracy_{num_labels}_{now}.png")


# Save model
savepath = f"/home/prashtata/gradschool/asl/trained_models/model_lstm_div_{num_labels}_03_12.pth"
torch.save(model.state_dict(), savepath)