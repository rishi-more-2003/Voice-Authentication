from preprocess_audio import pattern_generate
from preprocess_audio import pad_crop
import matplotlib.pyplot as plt
from model import predict
import numpy as np

params = {"N_FFT": 1024,
        "Mel_Dim": 80,
        "Frame_Length": 1024,
        "Frame_Shift": 256,
        "Sample_Rate": 22050,
        "Mel_F_Min": 0,
        "Mel_F_Max": 8000,}

path_1 = 'Rishi/audio_samples/HelloHello5.wav'
path_2 = 'Rishi/audio_samples/ThisIsRishi12345.wav'
audio, mel_1 = pattern_generate(path_1, params['N_FFT'], params['Mel_Dim'], params['Sample_Rate'], params['Frame_Shift'],
                                  params['Frame_Length'], params['Mel_F_Min'], params['Mel_F_Max'])
audio, mel_2 = pattern_generate(path_2, params['N_FFT'], params['Mel_Dim'], params['Sample_Rate'], params['Frame_Shift'],
                                  params['Frame_Length'], params['Mel_F_Min'], params['Mel_F_Max'])
mel_1 = pad_crop(mel_1, mode = 'mean')
mel_2 = pad_crop(mel_2, mode = 'mean')
mel_3 = np.load("Rishi/audio_samples/YoUgykRElWw_00012.npy")
mel_4 = np.load("Rishi/audio_samples/YoUgykRElWw_00013.npy")
mel_5 = np.load("Rishi/audio_samples/zo4Uv-dchPQ_00002.npy")
print(mel_3.shape)
# plt.matshow(mel)
# plt.show()
print(predict(mel_2,mel_1))
print(predict(mel_2,mel_3))
print(predict(mel_2,mel_4))
print(predict(mel_1,mel_3))
print(predict(mel_1,mel_5))
print(predict(mel_3,mel_4))
print(predict(mel_4,mel_5))
