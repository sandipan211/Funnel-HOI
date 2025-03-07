from .models_gen.gen_vlkt import build as build_gen
from .models_hoiclip.hoiclip import build as build_models_hoiclip
from .models_BTP.hoiclip import build as build_models_BTP
# from .models_BTP_akshit.hoiclip import build as build_models_BTP_25
from .models_BTP_paul.hoiclip import build as build_models_BTP_Paul25

from .visualization_hoiclip.gen_vlkt import build as visualization
from .visualization_hoiclip.gen_vlkt_demo import build as visualization_demo
from .generate_image_feature.generate_verb import build as generate_verb

from .models_BTP_paul.gen_vlkt import build as visualization_btp_paul25
from .models_BTP.gen_vlkt import build as visualization_btp_24

def build_model(args):
    if args.model_name == "HOICLIP":
        return build_models_hoiclip(args)
    elif args.model_name == "HOI_BTP24":
        return build_models_BTP(args)
    # elif args.model_name == "HOI_BTP25":
    #     return build_models_BTP_25(args)
    elif args.model_name == "HOI_BTP_PAUL25":
        return build_models_BTP_Paul25(args)
    elif args.model_name == "GEN":
        return build_gen(args)
    elif args.model_name == "VISUALIZATION":
        return visualization(args)
    elif args.model_name == "VISUALIZATION_DEMO":
        return visualization_demo(args)
    elif args.model_name == "VISUALIZATION_BTP_PAUL25":
        return visualization_btp_paul25(args)
    elif args.model_name == "VISUALIZATION_BTP_24":
        return visualization_btp_24(args)
    elif args.model_name == "GENERATE_VERB":
        return generate_verb(args)

    raise ValueError(f'Model {args.model_name} not supported')
