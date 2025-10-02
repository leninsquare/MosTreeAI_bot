from .augments import get_transform

class TreesDataset:
    def __init__(
        self,
        images,
        augments_type: str = "none"
    ):
        self.data = images
        self.transform = get_transform(augments_type)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)

        return image
