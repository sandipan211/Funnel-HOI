import torch
import argparse
from PIL import Image
from .tokenizer import SimpleTokenizer
import torchvision.transforms as transforms
import timm
from .models import SLIP


def get_state_dict(model_weight):
    state_dict = torch.load(model_weight)['state_dict']
    # print(state_dict.keys())
    state_dict_removed = {}
    for k, value in state_dict.items():
        if "module." in k:
            k_removed = k.split("module.")[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k] = value
    # print(state_dict_removed.keys())

    # check num_patches and understand which Vit base version - current SLIP is Vit-B/16
    # for k in state_dict_removed.keys():
    #     if 'patch_embed.proj.weight' in k:
    #         print(k, state_dict_removed[k].shape)
    return state_dict_removed

def get_transform(input_size=224):
    # from https://github.com/facebookresearch/SLIP/blob/c6faf5d03cbfa7d529d210779f859cd3dddec09a/main.py#L167

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])
    return transform

def load(model_name, model_weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'ViT-B/16':
        vision_model = timm.create_model('vit_base_patch16_224', num_classes=0) 
    else:
        vision_model = None
        raise NotImplementedError('This backbone not implemented yet/no weights available')
    model_slip = SLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, ssl_mlp_dim=4096, ssl_emb_dim=256)
    # print(model_slip)
    state_dict = get_state_dict(model_weight)
    miss, extra = model_slip.load_state_dict(state_dict, strict=False)
    # print(miss)
    # print('\n\n')
    # print(extra)
    model_slip.to(device)
    transform = get_transform()
    return model_slip, transform

def main(args):

    print('Modified SLIP (my version)')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load(args.model_name, args.model_weight)
    image = preprocess(Image.open("../HICO_test2015_00000001.jpg")).unsqueeze(0).to(device)
    tokenize = SimpleTokenizer()
    text = tokenize(["a photo of a horse"]).unsqueeze(0).to(device)
    print(text.shape)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        print(f'img ftrs = {image_features.shape}')
        print(f'txt_ftrs = {text_features.shape}')
        # print(f'contextual tokens = {ctx.shape}')



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ZeroShot")
    # parser.add_argument("--batch-size", default=128, type=int)
    # parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--model_name", default="ViT-B/32", help="Name of the vision backbone to use.")
    parser.add_argument("--model_weight", default="/home/gpuuser6/Sandipan/HOICLIP/ModifiedSLIP/slip_base_50ep.pt", type=str, help="pretrain model weight path.")
    # parser.add_argument("--input-size", default=224, type=int, help="Image resolution.")
    # parser.add_argument("--output-file", default="", type=str, help="output results file")

    args = parser.parse_args()
    main(args)