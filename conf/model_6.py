from conf import default, general, paths
from models.resunet import ResUnet

def get_model():
    print('Model RGBNir + int')
    lidar_bands = [4]
    input_depth_0 = general.N_OPTICAL_BANDS
    input_depth_1 = len(lidar_bands)
    #depths = [64, 128, 256, 512]
    depths = [32, 64, 128, 256]
    print(f'Model size: {depths}')
    print(f'Input shape: {input_depth_0}, {input_depth_1}')
    model = ResUnet(input_depth_0, input_depth_1, depths, general.N_CLASSES)

    return model, lidar_bands
