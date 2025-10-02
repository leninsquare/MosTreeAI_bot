import albumentations as A
from albumentations.pytorch import ToTensorV2

_AUGMENT_REGISTRY = {}

def register_augment(name: str):
    """Decorator to register a transform class or factory under a name."""
    def decorator(func_or_class):
        _AUGMENT_REGISTRY[name] = func_or_class
        return func_or_class
    return decorator

def get_transform(name: str, **kwargs):
    """Instantiate a transform by name."""
    if not name:
        raise ValueError("Transform name cannot be empty")

    parts = name.split('_')
    head, tail = parts[0], '_'.join(parts[1:]) if len(parts) > 1 else None

    if head not in _AUGMENT_REGISTRY:
        available = list(_AUGMENT_REGISTRY.keys())
        raise KeyError(f"Transform '{head}' not found. Available: {available}")

    factory = _AUGMENT_REGISTRY[head]

    if tail is None:
        return factory(**kwargs)

    return factory(tail, **kwargs)

@register_augment("none")
class NoAugment:
    '''
    A transform that is equivalent to T.ToTensor
    '''
    def __init__(self):
        self.transform = A.Compose([
            A.Normalize(mean=0, std=1), 
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform(image=image)["image"]

@register_augment("square")
class SquareAugment:
    '''
    A transform that is equivalent to T.ToTensor
    '''
    def __init__(self):
        IMG_SIZE = 256
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                border_mode=0),
            A.Normalize(mean=0, std=1), 
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform(image=image)["image"]
    