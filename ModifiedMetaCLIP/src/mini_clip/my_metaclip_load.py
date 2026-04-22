import torch
from . import create_model_and_transforms, get_tokenizer
# from src.mini_clip.tokenizer import tokenize
import argparse
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pprint import pprint

# def get_state_dict(model_weight):
#     state_dict = torch.load(model_weight)
#     pprint(state_dict.keys())
#     state_dict_removed = {}
#     for k, value in state_dict.items():
#         if "module." in k:
#             k_removed = k.split("module.")[-1]
#             state_dict_removed[k_removed] = value
#         else:
#             state_dict_removed[k] = value
#     pprint(state_dict_removed.keys())
#     return state_dict_removed


def load(model_name, model_weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_metaclip, prep_train, prep_test = create_model_and_transforms(model_name, pretrained=model_weight, device=device)
    # state_dict = get_state_dict(model_weight)
    # model_metaclip.load_state_dict(state_dict, strict=True)
    # model_metaclip.to(device)
    return model_metaclip, prep_train, prep_test

def main(args):

    print('Original MetaCLIP (my version)')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, val_preprocess = load(args.model_name, args.model_weight)
    image = val_preprocess(Image.open("/home/gpuuser6/Sandipan/HOICLIP/HICO_test2015_00000002.jpg")).unsqueeze(0).to(device)
    # tokenize = get_tokenizer("facebook/xlm-v-base")
    tokenize = get_tokenizer()
    text = tokenize(["building", "horse", "person", "car"]).to(device)
    print(text.shape)
    print(image.device, text.device)
    with torch.no_grad():
        image_features, ctx = model.encode_image(image)
        text_features = model.encode_text(text)
        print(f'img ftrs = {image_features.shape}')
        print(f'txt_ftrs = {text_features.shape}')
        print(f'ctx_ftrs = {ctx.shape}')

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
        

    # print(f'Similarity: {logits}')



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ZeroShot")
    # parser.add_argument("--batch-size", default=128, type=int)
    # parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--model_name", default="ViT-B-32-worldwide@WorldWideCLIP", help="Name of the vision backbone to use.")
    parser.add_argument("--model_weight", default="/home/gpuuser6/Sandipan/HOICLIP/MetaCLIP/metaclip2_b32_224px_worldwide.pt", type=str, help="pretrain model weight path.")
    # parser.add_argument("--input-size", default=224, type=int, help="Image resolution.")
    # parser.add_argument("--output-file", default="", type=str, help="output results file")

    args = parser.parse_args()
    main(args)