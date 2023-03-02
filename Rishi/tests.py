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

path_1 = 'Rishi\HelloHello5.wav'
path_2 = 'Rishi\ThisIsRishi12345.wav'
audio, mel_1 = pattern_generate(path_1, params['N_FFT'], params['Mel_Dim'], params['Sample_Rate'], params['Frame_Shift'],
                                  params['Frame_Length'], params['Mel_F_Min'], params['Mel_F_Max'])
audio, mel_2 = pattern_generate(path_2, params['N_FFT'], params['Mel_Dim'], params['Sample_Rate'], params['Frame_Shift'],
                                  params['Frame_Length'], params['Mel_F_Min'], params['Mel_F_Max'])
mel_1 = pad_crop(mel_1, mode = 'mean')
mel_2 = pad_crop(mel_2, mode = 'mean')
print(mel_1.shape)
# plt.matshow(mel)
# plt.show()
print(predict(mel_2,mel_1))
