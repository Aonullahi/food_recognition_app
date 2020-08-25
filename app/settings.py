import os

MODELS_ROOT = os.path.abspath(__file__ + "/../trained-models/")
DATA_ROOT = os.path.abspath(__file__ + "/../../data/raw")
LOG_FILE = os.path.abspath(__file__ + "/../../logs/application.log")
PORT = 8080
MODEL_DIR = os.path.abspath(__file__ + "/../trained-models/mask_rcnn_foodmodel_0030-v0.h5")

IMG_DIR = os.path.abspath(__file__ + "/../trained-models/test.jpeg")