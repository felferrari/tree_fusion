from conf import default, general, paths
from models.resunet import JointFusion

def get_model():
    print('Model RGBNir + nX nY')
    lidar_bands = [0, 1]
    input_depth_0 = general.N_OPTICAL_BANDS
    input_depth_1 = len(lidar_bands)
    #depths = [64, 128, 256, 512]
    depths = [32, 64, 128, 256]
    print(f'Model size: {depths}')
    print(f'Input shape: {input_depth_0}, {input_depth_1}')
    model = JointFusion(input_depth_0, input_depth_1, depths, general.N_CLASSES)

    return model, lidar_bands
