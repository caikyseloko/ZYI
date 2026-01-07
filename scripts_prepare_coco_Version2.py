#!/usr/bin/env python3
"""
Prepara COCO 2014 em 128x128 e cria annotations_processed.json (map filename -> caption).
Opcional: usa BLIP (via HuggingFace) para gerar captions alternativas.

Uso:
python scripts/prepare_coco.py --coco-ann /path/captions_train2014.json --images-dir /path/train2014 --out-dir data/coco128 --use-blip False
"""
import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

def load_coco_captions(coco_ann_path):
    with open(coco_ann_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    images = {im['id']: im for im in ann['images']}
    caps = defaultdict(list)
    for c in ann['annotations']:
        caps[c['image_id']].append(c['caption'])
    return images, caps

def save_image_resized(src_path, dst_path, size=(128,128)):
    with Image.open(src_path) as im:
        im = im.convert('RGB')
        im = im.resize(size, Image.BICUBIC)
        im.save(dst_path, quality=95)

def prepare(coco_ann, images_dir, out_dir, use_blip=False):
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    imgs_dir_out = os.path.join(out_dir, 'images')
    Path(imgs_dir_out).mkdir(exist_ok=True)
    images, caps = load_coco_captions(coco_ann)

    # optionally load BLIP (if requested)
    use_blip_model = False
    if use_blip:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            use_blip_model = True
        except Exception as e:
            print("BLIP unavailable, skipping BLIP captions:", e)
            use_blip_model = False

    annotations_out = {}
    skipped = 0
    for img_id, iminfo in tqdm(images.items(), desc="Processing images"):
        filename = iminfo['file_name']
        src = os.path.join(images_dir, filename)
        if not os.path.exists(src):
            skipped += 1
            continue
        dst_name = f"{img_id}.jpg"
        dst = os.path.join(imgs_dir_out, dst_name)
        try:
            save_image_resized(src, dst, size=(128,128))
        except Exception as e:
            print("Error processing", src, e)
            continue
        caption = caps.get(img_id, [""])[0]
        if use_blip_model:
            try:
                from PIL import Image as PILImage
                img_pil = PILImage.open(dst).convert('RGB')
                inputs = processor(images=img_pil, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_length=40)
                blip_caption = processor.decode(out[0], skip_special_tokens=True)
                caption = blip_caption
            except Exception:
                pass
        annotations_out[dst_name] = caption

    ann_out_path = os.path.join(out_dir, 'annotations_processed.json')
    with open(ann_out_path, 'w', encoding='utf-8') as f:
        json.dump(annotations_out, f, ensure_ascii=False, indent=2)
    print(f"Done. Images: {len(annotations_out)}. Skipped: {skipped}")
    print("Images in:", imgs_dir_out)
    print("Annotations saved to:", ann_out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--coco-ann', required=True, help='coco captions JSON (captions_train2014.json)')
    p.add_argument('--images-dir', required=True, help='path to train2014 or val2014 images')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--use-blip', action='store_true', help='use BLIP to regenerate captions')
    args = p.parse_args()
    prepare(args.coco_ann, args.images_dir, args.out_dir, use_blip=args.use_blip)