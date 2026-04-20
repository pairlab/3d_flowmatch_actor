from .rlbench import RLBenchDepth2Cloud
from .mesa70 import Mesa70PixelGridCloud


def fetch_depth2cloud(dataset_name, img_size=(256, 256)):
    dataset_name = dataset_name.lower()
    if 'mesa70' in dataset_name:
        return Mesa70PixelGridCloud(img_size)
    if 'mesa' in dataset_name:
        return RLBenchDepth2Cloud(img_size)
    if 'peract2' in dataset_name:
        return RLBenchDepth2Cloud(img_size)
    if 'rlbench' in dataset_name:
        return RLBenchDepth2Cloud(img_size)
    return None
