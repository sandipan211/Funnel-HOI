import torch
from . import create_model, factory, tokenize
import argparse
from PIL import Image

def get_state_dict(model_weight):
    state_dict = torch.load(model_weight)
    state_dict_removed = {}
    for k, value in state_dict.items():
        if "module." in k:
            k_removed = k.split("module.")[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k] = value
    return state_dict_removed

def get_transform(input_size=224):
    # original ALIP code demands two slightly different transforms for train and test - but we keep it as test mode always, since it then becomes identical to CLIP's transforms

    transform = factory.image_transform(input_size, 
                                        is_train=False,
                                        mean=(0.48145466, 0.4578275, 0.40821073),
                                        std= (0.26862954, 0.26130258, 0.27577711))
    return transform

def load(model_name, model_weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_alip = create_model(model_name)
    state_dict = get_state_dict(model_weight)
    model_alip.load_state_dict(state_dict, strict=True)
    model_alip.to(device)
    transform = get_transform()
    return model_alip, transform

def main(args):

    print('Modified ALIP (my version)')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load(args.model_name, args.model_weight)
    image = preprocess(Image.open("HICO_test2015_00000001.jpg")).unsqueeze(0).to(device)
    text = tokenize(["a photo of a horse"]).to(device)
    with torch.no_grad():
        image_features, ctx = model.encode_image(image)
        text_features = model.encode_text(text)
        print(f'img ftrs = {image_features.shape}')
        print(f'txt_ftrs = {text_features.shape}')
        print(f'contextual tokens = {ctx.shape}')



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ZeroShot")
    # parser.add_argument("--batch-size", default=128, type=int)
    # parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--model_name", default="ViT-B/32", help="Name of the vision backbone to use.")
    parser.add_argument("--model_weight", default="", type=str, help="pretrain model weight path.")
    # parser.add_argument("--input-size", default=224, type=int, help="Image resolution.")
    # parser.add_argument("--output-file", default="", type=str, help="output results file")

    args = parser.parse_args()
    main(args)