import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.bfloat16, attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def convert_img_features(img):
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=img, return_tensors="pt", padding=True)
    image_features = model.get_image_features(**inputs)
    return image_features[1]


