import tensorflow as tf
import tensorflow_hub as hub

import tensorflow as tf
import tensorflow_hub as hub

def create_model(num_classes=101, hub_url="https://tfhub.dev/deepmind/i3d-kinetics-400/1"):
   
    
    # Load the I3D model from TensorFlow Hub
    i3d = hub.load(hub_url).signatures['default']
    
    
    
    
    return i3d
