# ----------------------------------------------------------------------------------------------------------------------
# /*==========================================================*\
# |     ||  -          Copyright Piovesan            - ||      |
# |     ||  -       Audio Features Extraction        - ||      |
# |     ||  -          By: Thiago Piovesan           - ||      |
# |     ||  -          Versao atual: 1.0.0           - ||      |
# \*==========================================================*/
#   This software is confidential and property of NoLeak.
#   Your distribution or disclosure of your content is not permitted without express permission from NoLeak.
#   This file contains proprietary information.

#   Link do Github: https://github.com/ThiagoPiovesan
# ----------------------------------------------------------------------------------------------------------------------
# Libs Importation:
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from urbansounddataset import UrbanSoundDataset
from cnn import CNNetwork
# from torchvision import datasets
# from torchvision.transforms import ToTensor

# 1 - Donwload dataset
# 2 - Create data loader
# 3 - Build model
# 4 - Train
# 5 - Save trained model

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001
ANNOTATIONS_FILE: str = "C:/Users/thiag/Datasets/UrbanSound8K/metadata/UrbanSound8k.csv"
AUDIO_DIR: str = "C:/Users/thiag/Datasets/UrbanSound8K/audio"
SAMPLE_RATE: int = 22050
NUM_SAMPLES: int = 22050

# class FeedForwardNet(nn.Module):
    
#     def __init__(self) -> None:
#         super().__init__()
        
#         self.flatten = nn.Flatten()
#         self.dense_layers = nn.Sequential(
#             nn.Linear(28*28, 256),
#             nn.ReLU(),
#             nn.Linear(256, 10)
#         )
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input_data):
#         flattened_data = self.flatten(input_data)
#         logits = self.dense_layers(flattened_data)
#         predictions = self.softmax(logits)

#         return predictions
        
# def donwload_mnist_dataset():
#     train_data = datasets.MNIST(
#         root="AudioFeatExtract/Training a feed forward network/data",
#         download=True,
#         train=True,
#         transform=ToTensor()
#     )
#     validation_data = datasets.MNIST(
#         root="AudioFeatExtract/Training a feed forward network/data",
#         download=True,
#         train=False,
#         transform=ToTensor()
#     )

#     return train_data, validation_data

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        
        # Backpropagate loss and upadte weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    print("Loss: {}".format(loss.item()))
        
def train(model, data_loader, loss_fn, optimiser, device, epochs):

    for i in range(epochs):
        print("Epoch: {}".format(i+1))
            
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("----------------------")
    
    print("Training is done.")

if __name__ == "__main__":
    # Building the model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Using {} device".format(device))
        
    # Instantiate our Dataset object and Create a data loader
    mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, device)
    
    train_data_loader = create_data_loader(usd, BATCH_SIZE)
    
    cnn = CNNetwork().to(device)
    print(cnn)
    
    # Instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr = LEARNING_RATE)
    
    # Train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    
    # Save the model
    torch.save(cnn.state_dict(), "feedforwardnet_cnn.pth")
    print("Model trained and stored at feedforwardnet_cnn.pth")