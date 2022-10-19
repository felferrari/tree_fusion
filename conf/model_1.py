from conf import default, general, paths
from models.resunet import ResUnetOpt

def get_model():
    print('Model RGBNir')
    lidar_bands = None
    input_depth_0 = general.N_OPTICAL_BANDS
    input_depth_1 = 0
    #depths = [64, 128, 256, 512]
    depths = [32, 64, 128, 256]
    print(f'Model size: {depths}')
    print(f'Input shapes: {input_depth_0}, {input_depth_1}')
    model = ResUnetOpt(input_depth_0, depths, general.N_CLASSES)

    return model, lidar_bands
