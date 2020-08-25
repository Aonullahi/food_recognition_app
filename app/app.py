import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import skimage

from flask import Flask, request, jsonify, make_response, render_template, url_for
from settings import *
import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve()
#ROOT_DIR = os.path.abspath("")
print(PACKAGE_ROOT)
print(MODEL_DIR)

# Import Mask RCNN
sys.path.append(PACKAGE_ROOT)  # To find local version of the library

app = Flask(__name__)

#####################################################################################
################################### ROUTES ##########################################
#####################################################################################

from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return ("OK", 200)

class FoodModelConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "foodmodel"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # background + 1 (classes of food)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = FoodModelConfig()
config.display()

class InferenceConfig(FoodModelConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85
    
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


model.load_weights(MODEL_DIR, by_name=True) 


class_names = ['BG', 'Chicken', 'Eba', 'Fish', 'Rice', 'Bread' ]


#for image_path in image_paths:
img = skimage.io.imread(IMG_DIR)
img_arr = np.array(img)
results = model.detect([img_arr], verbose=1)
r = results[0]
#visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
 #                           class_names, r['scores'], figsize=(5,5))

#ans = visualize.apply_mask(img, r['masks'], color=None, alpha=0.5)
print(r['class_ids'])



@app.route('/predict',methods=['POST', 'GET'])
def predict():
    pass

@app.route('/predict_api', methods=['POST'])
def predict_api():
    pass


#####################################################################################
############################### INTIALIZATION CODE ##################################
#####################################################################################

if __name__ == '__main__':
    try:
        port = int(PORT)
    except Exception as e:
        print("Failed to bind to port {}".format(PORT))
        port = 80

    app.run(port=port , debug = True)

    # disable logging so it doesn't interfere with testing
    app.logger.disabled = True
