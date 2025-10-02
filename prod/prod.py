from .inference import inference

MODEL_PATH = 'model/prod/models/'

MODEL_INFO = {
    'species': ('trees.pth', 10, None),
    'mechanical': ('mechanical.pth', 2, 0.09),
    'hollow': ('hollow.pth', 2, 0.03),
    'rot': ('rot.pth', 2, 0.45),
    'cracks': ('cracks.pth', 2, 0.31),
    'em_slope': ('em_slope.pth', 2, 0.57),
    'kappa': ('kappa.pth', 2, 0.33),
}

def analyze(images, device):
    '''
    images are HWC, RGB, uint8
    '''
    results = {}

    for key in MODEL_INFO:
        model_name, num_classes, threshold = MODEL_INFO[key]
        results[key] = inference(images, num_classes, device, MODEL_PATH + model_name, 'square', thr=threshold)

    return results
