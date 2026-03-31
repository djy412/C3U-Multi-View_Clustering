# Parameter configurations
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("C3U_DATA_ROOT", PROJECT_ROOT / "data"))
WEIGHTS_ROOT = Path(os.getenv("C3U_WEIGHTS_ROOT", PROJECT_ROOT / "weights"))

BATCH_SIZE = 256
WORKERS = 8         #--- Load data in parallel by choosing the best num of workers for your system
PRE_TRAIN_EPOCHS = 100
FINE_TUNE_EPOCHS = 30
MODEL_FILENAME = './data/model.ckpt'

LR = 0.001              #--- Learning rate 
LAMBDA = 0.5             #--- Common Contrastive loss
MARGIN = 0.5            #--- Complementary Margin 
GAMMA = 0.6             #--- Complementary Clustering loss
TEMPERATURE_CON = 0.3   #--- Controls the stage I contrastive loss
W_PULL = 0.6            #--- Controls the stage II soft assignment loss

BETA = 0.0
BETA_IB = 0.01      #--- New param for IB KL weight; add to config

NORMALIZED = True
LATENT_DIM = 3
LATENT_DIM_C = 18
LATENT_DIM_U = 18
NUM_CLASSES = 10    #--- 1446 for CelebA full
TOLERANCE = 0.01    #--- How close to the last estimage is good enough
UPDATE_INTERVAL = 3 #--- How often to update the estimated "true data", 1 would = updating every Epoch

#dataset_name = 'MULTI-USPS'        # 32x32
#dataset_name = 'MULTI-MNIST'       # 32x32
#dataset_name = 'MULTI-FASHION'     # 32x32
#dataset_name = 'Fashion'           # 32x32
#dataset_name = 'MULTI_COIL_10'     # 64x64
#dataset_name = 'MULTI-MVP-N'       # 96x96
#dataset_name = 'MULTI_COIL_20'     # 64x64
dataset_name = 'MULTI_STL-10'      # 96x96
#dataset_name = 'MULTI_Eglin'       # 96x96
#dataset_name = 'MULTI_KITTI_00'    # 128x128
#dataset_name = 'MULTI-CIFAR-10'     # 32x32
#dataset_name = 'Caltech_2V'        # multiple
#dataset_name = 'Caltech_3V'        # multiple
#dataset_name = 'Caltech101_7'      # multiple
#dataset_name = 'MULTI-CelebA'      # 128x128

views = 3
CHANNELS = 3
IMG_SIZE = 96  # must match dataset images
IHMC = False