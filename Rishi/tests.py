from preprocess_audio import get_mel
import matplotlib.pyplot as plt
from model import get_prediction
import numpy as np

path_1 = 'audio_samples/HelloHello5.wav'
path_2 = 'audio_samples/ThisIsRishi12345.wav'
mel_1 = get_mel(path_1)
mel_2 = get_mel(path_2)
mel_3 = np.load("audio_samples/YoUgykRElWw_00012.npy")
mel_4 = np.load("audio_samples/YoUgykRElWw_00013.npy")
mel_5 = np.load("audio_samples/zo4Uv-dchPQ_00002.npy")
print(mel_3.shape)
print(get_prediction(mel_2,mel_1))
print(get_prediction(mel_2,mel_3))
print(get_prediction(mel_2,mel_4))
print(get_prediction(mel_1,mel_3))
print(get_prediction(mel_1,mel_5))
print(get_prediction(mel_3,mel_4))
print(get_prediction(mel_4,mel_5))


