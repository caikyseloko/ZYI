#!/usr/bin/env python3
"""
Prepara COCO 2014 em 128x128 e cria annotations_processed.json (map filename -> caption).
Aceita:
  --use-blip true|false    : habilita/desabilita geração de captions com BLIP (default: false)
  --max-images N           : processa no máximo N imagens (default: 2000)

Uso:
python scripts/prepare_coco.py \
  --coco-ann /path/captions_train2014.json \
  --images-dir /path/train2014 \
  --out-dir data/coco128 \
  --use-blip false \
  --max-images 2000
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

def prepare(coco_ann, images_dir, out_dir, use_blip=False, max_images=2000):
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
            print("BLIP model loaded on", device)
        except Exception as e:
            print("BLIP unavailable, continuing without BLIP captions. Error:", e)
            use_blip_model = False

    annotations_out = {}
    skipped = 0
    processed = 0
    total_images = len(images)
    # iterate deterministically by image id order to make limit predictable
    for img_idx, (img_id, iminfo) in enumerate(tqdm(sorted(images.items()), desc="Processing images")):
        if processed >= max_images:
            break
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
            skipped += 1
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
            except Exception as e:
                print("BLIP captioning failed for", dst_name, ":", e)
        annotations_out[dst_name] = caption
        processed += 1

    ann_out_path = os.path.join(out_dir, 'annotations_processed.json')
    with open(ann_out_path, 'w', encoding='utf-8') as f:
        json.dump(annotations_out, f, ensure_ascii=False, indent=2)
    print(f"Done. Processed: {processed}. Skipped (missing/corrupt): {skipped}. Total in COCO file: {total_images}")
    print("Images in:", imgs_dir_out)
    print("Annotations saved to:", ann_out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--coco-ann', required=True, help='coco captions JSON (captions_train2014.json)')
    p.add_argument('--images-dir', required=True, help='path to train2014 or val2014 images')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--use-blip', type=str, default='false', choices=['true','false'],
                   help='true or false to enable BLIP caption regeneration (default: false)')
    p.add_argument('--max-images', type=int, default=2000,
                   help='maximum number of images to process (default: 2000)')
    args = p.parse_args()
    use_blip_flag = args.use_blip.lower() == 'true'
    prepare(args.coco_ann, args.images_dir, args.out_dir, use_blip=use_blip_flag, max_images=args.max_images)
