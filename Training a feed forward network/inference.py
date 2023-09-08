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
from cnn import CNNetwork
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

# ----------------------------------------------------------------------------------------------------------------------
class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]
# ----------------------------------------------------------------------------------------------------------------------

def predict(model, input, target, class_mapping):
    model.eval()        # Swtich when you need to evaluate -> train() to switch to train mode
    
    with torch.no_grad(): # We are evaluating the model
        predictions = model(input)

        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ] -> SUM = 1
        # print("The predictions Tensor is: {}".format(predictions))
        
        predictions_index = predictions[0].argmax(0)
        predicted = class_mapping[predictions_index]
        
        expected = class_mapping[target]
        
    return predicted, expected
        
if __name__ == "__main__":
    # 1 - load back the model
    cnn = CNNetwork()
    state_dict = torch.load("cnnnet.pth")
    cnn.load_state_dict(state_dict)
    
    # 2 - load Urban Sound dataset
    mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, "cpu")
    
    # 3 - get a sample from the Urban Sound dataset for inference
    input, target = usd[0][0], usd[0][1] # Tensor with 3 dimensions [num_channels, Frequency Axes, Time axes] -> Need to add a batch size as first parameter
    input.unsqueeze_(0)  # Que underscore allows to add it inplace.
       
    # 4 - make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)
    
    print("Predicted: {}, Expected: {}".format(predicted, expected))