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
from torch import nn
from torchsummary import summary

class CNNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        # 4 Conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # = Gray scale image, channels = 3 if RGB
                out_channels=16,    # 16 filters will be applied
                kernel_size=3,      # Valor meio que comum
                stride=1,
                padding=2
            ),
            nn.ReLU(),              # Função de ativação
            nn.MaxPool2d(kernel_size=2)
        )
        # Block 2:
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,     # Equals to the output of the previous layer
                out_channels=32,    # 32 filters will be applied
                kernel_size=3,      # Valor meio que comum
                stride=1,
                padding=2
            ),
            nn.ReLU(),              # Função de ativação
            nn.MaxPool2d(kernel_size=2)
        )
        # Block 3:
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,      # = Gray scale image, channels = 3 if RGB
                out_channels=64,    # 16 filters will be applied
                kernel_size=3,      # Valor meio que comum
                stride=1,
                padding=2
            ),
            nn.ReLU(),              # Função de ativação
            nn.MaxPool2d(kernel_size=2)
        )
        # Block 4:
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,      # = Gray scale image, channels = 3 if RGB
                out_channels=128,    # 16 filters will be applied
                kernel_size=3,      # Valor meio que comum
                stride=1,
                padding=2
            ),
            nn.ReLU(),              # Função de ativação
            nn.MaxPool2d(kernel_size=2)
        )
        # Flatten:
        self.flatten = nn.Flatten()
        
        # Dense layer = Linear
        self.linear = nn.Linear(128 * 5 * 4, 10)
        
        # Softmax layer = result
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):
        """
            How to pass data from one layer to another.
        """
        # Camadas Convolucionais
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten Layer
        x = self.flatten(x)
        
        # Linear Layer
        logits = self.linear(x)
        
        # make predictions
        predictions = self.softmax(logits)
        
        return predictions
    
if __name__ == '__main__':
    cnn = CNNetwork()
    
    summary(cnn.cuda(), (1, 64, 44))