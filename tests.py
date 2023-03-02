from preprocess_audio import pattern_generate
from preprocess_audio import pad_crop
import matplotlib.pyplot as plt
from model import predict

params = {"N_FFT": 1024,
        "Mel_Dim": 80,
        "Frame_Length": 1024,
        "Frame_Shift": 256,
        "Sample_Rate": 22050,
        "Mel_F_Min": 0,
        "Mel_F_Max": 8000,}

path = 'VBA\HelloHello5.wav'
audio, mel = pattern_generate(path, params['N_FFT'], params['Mel_Dim'], params['Sample_Rate'], params['Frame_Shift'],
                                  params['Frame_Length'], params['Mel_F_Min'], params['Mel_F_Max'])
mel = pad_crop(mel, mode = 'mean')
print(mel.shape)
plt.matshow(mel)
plt.show()

print(predict(mel,mel))
