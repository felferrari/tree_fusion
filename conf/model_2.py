from conf import default, general, paths
from models.resunet import ResUnetOpt

def get_model():
    print('Model RGBNir')
    input_depth = general.N_OPTICAL_BANDS# + general.N_LIDAR_BANDS
    #depths = [64, 128, 256, 512]
    depths = [32, 64, 128, 256]
    print(f'Model size: {depths}')
    model = ResUnetOpt(input_depth, depths, general.N_CLASSES)

    return model
