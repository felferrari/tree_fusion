from conf import default, general, paths
from models.resunet import ResUnet

def get_model():

    input_depth = general.N_OPTICAL_BANDS + general.N_LIDAR_BANDS
    #depths = [64, 128, 256, 512]
    depths = [32, 64, 128, 256]
    model = ResUnet(input_depth, depths, general.N_CLASSES)

    return model
