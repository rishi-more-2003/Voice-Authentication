from preprocess_audio import get_mel
# from preprocess_audio import pad_crop
import matplotlib.pyplot as plt
from model import predict
import numpy as np

path_1 = 'Rishi/audio_samples/HelloHello5.wav'
path_2 = 'Rishi/audio_samples/ThisIsRishi12345.wav'
mel_1 = get_mel(path_1)
mel_2 = get_mel(path_2)
mel_3 = np.load("Rishi/audio_samples/YoUgykRElWw_00012.npy")
mel_4 = np.load("Rishi/audio_samples/YoUgykRElWw_00013.npy")
mel_5 = np.load("Rishi/audio_samples/zo4Uv-dchPQ_00002.npy")
print(mel_3.shape)
print(predict(mel_2,mel_1))
print(predict(mel_2,mel_3))
print(predict(mel_2,mel_4))
print(predict(mel_1,mel_3))
print(predict(mel_1,mel_5))
print(predict(mel_3,mel_4))
print(predict(mel_4,mel_5))
