import os, json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.radar_maps import rasterize

class FusionDataset(Dataset):
    """image (3,360,640) in [0,1]; radar (3,360,640) = D,V,M; labels (n,5) = cls,cx,cy,w,h normalised; token.
    corrupt: optional function acting on the radar point array before rasterising."""
    def __init__(self, root, split, corrupt=None):
        self.root, self.split, self.corrupt = root, split, corrupt
        self.items = [i for i in json.load(open(os.path.join(root, "index.json"))) if i["split"] == split]
        z = np.load(os.path.join(root, "radar_points.npz"))
        self.points = {k: z[k] for k in z.files}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tok = self.items[i]["token"]
        img = np.asarray(Image.open(os.path.join(self.root, "images", self.split, tok + ".jpg")),
                         dtype=np.float32) / 255.0
        P = self.points[tok]
        if self.corrupt is not None:
            P = self.corrupt(P)
        radar = rasterize(P)
        with open(os.path.join(self.root, "labels", self.split, tok + ".txt")) as f:
            rows = [list(map(float, l.split())) for l in f if l.strip()]
        lab = np.array(rows, np.float32).reshape(-1, 5)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(radar), torch.from_numpy(lab), tok

def collate(batch):
    imgs, radars, labs, toks = zip(*batch)
    return torch.stack(imgs), torch.stack(radars), list(labs), list(toks)