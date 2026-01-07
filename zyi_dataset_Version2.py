import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class COCO128Dataset(Dataset):
    """
    Estrutura esperada:
    out_dir/
      images/  (arquivos <image_id>.jpg)
      annotations_processed.json  (map filename -> caption)
    """
    def __init__(self, out_dir, transform=None, max_items=None):
        ann_path = os.path.join(out_dir, 'annotations_processed.json')
        if not os.path.exists(ann_path):
            raise FileNotFoundError("annotations_processed.json not found in " + out_dir)
        with open(ann_path, 'r', encoding='utf-8') as f:
            self.ann = json.load(f)
        self.images_dir = os.path.join(out_dir, 'images')
        self.filenames = list(self.ann.keys())
        if max_items:
            self.filenames = self.filenames[:max_items]
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img = Image.open(os.path.join(self.images_dir, fn)).convert('RGB')
        img = self.transform(img)
        caption = self.ann[fn]
        return img, caption