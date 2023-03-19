import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Layer

# Siamese L1 Distance class
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

model_path = "Rishi/100K_sample_siam_model_900K.h5"
model = keras.models.load_model(model_path,custom_objects={'L1Dist': L1Dist})

def predict(mel1, mel2):
    mel1 = mel1.reshape(-1,80,450)
    mel2 = mel2.reshape(-1,80,450)
    return model.predict([mel1,mel2])

if "__main__" == __name__:
    print("Model loaded successfully!")
    print(model.summary())