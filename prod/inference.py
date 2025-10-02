import torch
from torch.utils.data import DataLoader

from .modules import ResNetClassifier
from .dataset import TreesDataset

images = ""

MAP = {
    0: "ива",
    1: "ясенелистный клен",
    2: "тополь",
    3: "клен",
    4: "липа",
    5: "береза",
    6: "сосна",
    7: "ель",
    8: "другие деревья",
    9: "другие кустарники",
}

def inference(images, num_classes, device, model_path, augment_type, batch_size=8, thr=0.5):
    net = ResNetClassifier(num_classes if num_classes > 2 else 1).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()


    dataset = TreesDataset(
        images, 
        augments_type=augment_type
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if num_classes > 2:
                output = net(batch)
                output = output.argmax(dim=1)
                curr_pred = output.detach().cpu().tolist()
                y_pred.extend([MAP[i] for i in curr_pred])
            else:
                logits = net(batch)
                probs = torch.sigmoid(logits.squeeze(1))
                preds = (probs >= thr).long()
                curr_pred = preds.detach().cpu().tolist()
                y_pred.extend(curr_pred)

    return y_pred
    