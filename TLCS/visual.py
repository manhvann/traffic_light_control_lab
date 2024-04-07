import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load the model from the h5 file
model = load_model('D:\\Deep-QLearning-Agent-for-Traffic-Signal-Control\\TLCS\models\\model_5\\trained_model.h5')

# Save the model structure as a PNG file
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)