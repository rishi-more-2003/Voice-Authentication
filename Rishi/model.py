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

model_path = "saimese_2_conv_v5.h5"
model = keras.models.load_model(model_path,custom_objects={'L1Dist': L1Dist})
embedder = keras.models.load_model('embedder_head_1M.h5')
tail = keras.models.load_model('tail_1M.h5')

def generate_embedding(mel):
    return embedder.predict(mel.reshape(-1,80,450))

def calc_dist(input_embedding,validation_embedding):
    '''
    This method calculates the L1 distance between two embeddings
    '''
    dist = tf.math.abs(input_embedding - validation_embedding)
    return tail.predict(dist)

def get_prediction(mel_1,mel_2):
    '''
    This method returns the prediction of the model
    '''
    
    embed_1 = generate_embedding(mel_1)
    embed_2 = generate_embedding(mel_2)
    return calc_dist(embed_1,embed_2)
    

def predict(mel1, mel2):
    mel1 = mel1.reshape(-1,80,450)
    mel2 = mel2.reshape(-1,80,450)
    return model.predict([mel1,mel2])

if "__main__" == __name__:
    print("Model loaded successfully!")
    print(model.summary())