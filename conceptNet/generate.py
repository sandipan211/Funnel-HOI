# please run on google colab

import requests
import torch
# from ..datasets.hico_text_label import hico_obj_text_label, hico_verb_text_label 
hico_obj_text_label = [(0, 'a photo of a person'), (1, 'a photo of a bicycle'), (2, 'a photo of a car'),
                       (3, 'a photo of a motorcycle'), (4, 'a photo of an airplane'), (5, 'a photo of a bus'),
                       (6, 'a photo of a train'), (7, 'a photo of a truck'), (8, 'a photo of a boat'),
                       (9, 'a photo of a traffic light'), (10, 'a photo of a fire hydrant'),
                       (11, 'a photo of a stop sign'), (12, 'a photo of a parking meter'), (13, 'a photo of a bench'),
                       (14, 'a photo of a bird'), (15, 'a photo of a cat'), (16, 'a photo of a dog'),
                       (17, 'a photo of a horse'), (18, 'a photo of a sheep'), (19, 'a photo of a cow'),
                       (20, 'a photo of an elephant'), (21, 'a photo of a bear'), (22, 'a photo of a zebra'),
                       (23, 'a photo of a giraffe'), (24, 'a photo of a backpack'), (25, 'a photo of a umbrella'),
                       (26, 'a photo of a handbag'), (27, 'a photo of a tie'), (28, 'a photo of a suitcase'),
                       (29, 'a photo of a frisbee'), (30, 'a photo of a skis'), (31, 'a photo of a snowboard'),
                       (32, 'a photo of a sports ball'), (33, 'a photo of a kite'), (34, 'a photo of a baseball bat'),
                       (35, 'a photo of a baseball glove'), (36, 'a photo of a skateboard'),
                       (37, 'a photo of a surfboard'), (38, 'a photo of a tennis racket'), (39, 'a photo of a bottle'),
                       (40, 'a photo of a wine glass'), (41, 'a photo of a cup'), (42, 'a photo of a fork'),
                       (43, 'a photo of a knife'), (44, 'a photo of a spoon'), (45, 'a photo of a bowl'),
                       (46, 'a photo of a banana'), (47, 'a photo of an apple'), (48, 'a photo of a sandwich'),
                       (49, 'a photo of an orange'), (50, 'a photo of a broccoli'), (51, 'a photo of a carrot'),
                       (52, 'a photo of a hot dog'), (53, 'a photo of a pizza'), (54, 'a photo of a donut'),
                       (55, 'a photo of a cake'), (56, 'a photo of a chair'), (57, 'a photo of a couch'),
                       (58, 'a photo of a potted plant'), (59, 'a photo of a bed'), (60, 'a photo of a dining table'),
                       (61, 'a photo of a toilet'), (62, 'a photo of a tv'), (63, 'a photo of a laptop'),
                       (64, 'a photo of a mouse'), (65, 'a photo of a remote'), (66, 'a photo of a keyboard'),
                       (67, 'a photo of a cell phone'), (68, 'a photo of a microwave'), (69, 'a photo of an oven'),
                       (70, 'a photo of a toaster'), (71, 'a photo of a sink'), (72, 'a photo of a refrigerator'),
                       (73, 'a photo of a book'), (74, 'a photo of a clock'), (75, 'a photo of a vase'),
                       (76, 'a photo of a scissors'), (77, 'a photo of a teddy bear'), (78, 'a photo of a hair drier'),
                       (79, 'a photo of a toothbrush'), (80, 'a photo of nothing')]
hico_verb_text_label={
    0: 'a photo of a person adjusting',
    1: 'a photo of a person assembling',
    2: 'a photo of a person blocking',
    3: 'a photo of a person blowing',
    4: 'a photo of a person boarding',
    5: 'a photo of a person breaking',
    6: 'a photo of a person brushing with',
    7: 'a photo of a person buying',
    8: 'a photo of a person carrying',
    9: 'a photo of a person catching',
    10: 'a photo of a person chasing',
    11: 'a photo of a person checking',
    12: 'a photo of a person cleaning',
    13: 'a photo of a person controlling',
    14: 'a photo of a person cooking',
    15: 'a photo of a person cutting',
    16: 'a photo of a person cutting with',
    17: 'a photo of a person directing',
    18: 'a photo of a person dragging',
    19: 'a photo of a person dribbling',
    20: 'a photo of a person drinking with',
    21: 'a photo of a person driving',
    22: 'a photo of a person drying',
    23: 'a photo of a person eating',
    24: 'a photo of a person eating at',
    25: 'a photo of a person exiting',
    26: 'a photo of a person feeding',
    27: 'a photo of a person filling',
    28: 'a photo of a person flipping',
    29: 'a photo of a person flushing',
    30: 'a photo of a person flying',
    31: 'a photo of a person greeting',
    32: 'a photo of a person grinding',
    33: 'a photo of a person grooming',
    34: 'a photo of a person herding',
    35: 'a photo of a person hitting',
    36: 'a photo of a person holding',
    37: 'a photo of a person hopping on',
    38: 'a photo of a person hosing',
    39: 'a photo of a person hugging',
    40: 'a photo of a person hunting',
    41: 'a photo of a person inspecting',
    42: 'a photo of a person installing',
    43: 'a photo of a person jumping',
    44: 'a photo of a person kicking',
    45: 'a photo of a person kissing',
    46: 'a photo of a person lassoing',
    47: 'a photo of a person launching',
    48: 'a photo of a person licking',
    49: 'a photo of a person lying on',
    50: 'a photo of a person lifting',
    51: 'a photo of a person lighting',
    52: 'a photo of a person loading',
    53: 'a photo of a person losing',
    54: 'a photo of a person making',
    55: 'a photo of a person milking',
    56: 'a photo of a person moving',
    57: 'a photo of a person and',
    58: 'a photo of a person opening',
    59: 'a photo of a person operating',
    60: 'a photo of a person packing',
    61: 'a photo of a person painting',
    62: 'a photo of a person parking',
    63: 'a photo of a person paying',
    64: 'a photo of a person peeling',
    65: 'a photo of a person petting',
    66: 'a photo of a person picking',
    67: 'a photo of a person picking up',
    68: 'a photo of a person pointing',
    69: 'a photo of a person pouring',
    70: 'a photo of a person pulling',
    71: 'a photo of a person pushing',
    72: 'a photo of a person racing',
    73: 'a photo of a person reading',
    74: 'a photo of a person releasing',
    75: 'a photo of a person repairing',
    76: 'a photo of a person riding',
    77: 'a photo of a person rowing',
    78: 'a photo of a person running',
    79: 'a photo of a person sailing',
    80: 'a photo of a person scratching',
    81: 'a photo of a person serving',
    82: 'a photo of a person setting',
    83: 'a photo of a person shearing',
    84: 'a photo of a person signing',
    85: 'a photo of a person sipping',
    86: 'a photo of a person sitting at',
    87: 'a photo of a person sitting on',
    88: 'a photo of a person sliding',
    89: 'a photo of a person smelling',
    90: 'a photo of a person spinning',
    91: 'a photo of a person squeezing',
    92: 'a photo of a person stabbing',
    93: 'a photo of a person standing on',
    94: 'a photo of a person standing under',
    95: 'a photo of a person sticking',
    96: 'a photo of a person stirring',
    97: 'a photo of a person stopping at',
    98: 'a photo of a person straddling',
    99: 'a photo of a person swinging',
    100: 'a photo of a person tagging',
    101: 'a photo of a person talking on',
    102: 'a photo of a person teaching',
    103: 'a photo of a person texting on',
    104: 'a photo of a person throwing',
    105: 'a photo of a person tying',
    106: 'a photo of a person toasting',
    107: 'a photo of a person training',
    108: 'a photo of a person turning',
    109: 'a photo of a person typing on',
    110: 'a photo of a person walking',
    111: 'a photo of a person washing',
    112: 'a photo of a person watching',
    113: 'a photo of a person waving',
    114: 'a photo of a person wearing',
    115: 'a photo of a person wielding',
    116: 'a photo of a person zipping'
}

vcoco_obj_text_label_our = [(0, 'a photo of a person'), (1, 'a photo of a bicycle'),
                        (2, 'a photo of a car'), (3, 'a photo of a motorcycle'),
                        (4, 'a photo of an airplane'), (5, 'a photo of a bus'),
                        (6, 'a photo of a train'), (7, 'a photo of a truck'),
                        (8, 'a photo of a boat'), (9, 'a photo of a traffic light'),
                        (10, 'a photo of a fire hydrant'), (11, 'a photo of a stop sign'),
                        (12, 'a photo of a parking meter'), (13, 'a photo of a bench'),
                        (14, 'a photo of a bird'), (15, 'a photo of a cat'),
                        (16, 'a photo of a dog'), (17, 'a photo of a horse'),
                        (18, 'a photo of a sheep'), (19, 'a photo of a cow'),
                        (20, 'a photo of an elephant'), (21, 'a photo of a bear'),
                        (22, 'a photo of a zebra'), (23, 'a photo of a giraffe'),
                        (24, 'a photo of a backpack'), (25, 'a photo of a umbrella'),
                        (26, 'a photo of a handbag'), (27, 'a photo of a tie'),
                        (28, 'a photo of a suitcase'), (29, 'a photo of a frisbee'),
                        (30, 'a photo of a skis'), (31, 'a photo of a snowboard'),
                        (32, 'a photo of a sports ball'), (33, 'a photo of a kite'),
                        (34, 'a photo of a baseball bat'),
                        (35, 'a photo of a baseball glove'),
                        (36, 'a photo of a skateboard'), (37, 'a photo of a surfboard'),
                        (38, 'a photo of a tennis racket'), (39, 'a photo of a bottle'),
                        (40, 'a photo of a wine glass'), (41, 'a photo of a cup'),
                        (42, 'a photo of a fork'), (43, 'a photo of a knife'),
                        (44, 'a photo of a spoon'), (45, 'a photo of a bowl'),
                        (46, 'a photo of a banana'), (47, 'a photo of an apple'),
                        (48, 'a photo of a sandwich'), (49, 'a photo of an orange'),
                        (50, 'a photo of a broccoli'), (51, 'a photo of a carrot'),
                        (52, 'a photo of a hot dog'), (53, 'a photo of a pizza'),
                        (54, 'a photo of a donut'), (55, 'a photo of a cake'),
                        (56, 'a photo of a chair'), (57, 'a photo of a couch'),
                        (58, 'a photo of a potted plant'), (59, 'a photo of a bed'),
                        (60, 'a photo of a dining table'), (61, 'a photo of a toilet'),
                        (62, 'a photo of a tv'), (63, 'a photo of a laptop'),
                        (64, 'a photo of a mouse'), (65, 'a photo of a remote'),
                        (66, 'a photo of a keyboard'), (67, 'a photo of a cell phone'),
                        (68, 'a photo of a microwave'), (69, 'a photo of an oven'),
                        (70, 'a photo of a toaster'), (71, 'a photo of a sink'),
                        (72, 'a photo of a refrigerator'), (73, 'a photo of a book'),
                        (74, 'a photo of a clock'), (75, 'a photo of a vase'),
                        (76, 'a photo of a scissors'), (77, 'a photo of a teddy bear'),
                        (78, 'a photo of a hair drier'), (79, 'a photo of a toothbrush'),
                        (80, 'a photo of a person only'), (81, 'a photo of nothing')]

vcoco_verb_text_label={
    0: 'a photo of a person holding',
    1: 'a photo of a person standing',
    2: 'a photo of a person sitting',
    3: 'a photo of a person riding',
    4: 'a photo of a person walking',
    5: 'a photo of a person looking at',
    6: 'a photo of a person hitting with',
    7: 'a photo of a person hitting',
    8: 'a photo of a person eating',
    9: 'a photo of a person eating with',
    10: 'a photo of a person jumping',
    11: 'a photo of a person laying',
    12: 'a photo of a person talking on',
    13: 'a photo of a person carrying',
    14: 'a photo of a person throwing',
    15: 'a photo of a person catching',
    16: 'a photo of a person cutting with',
    17: 'a photo of a person cutting',
    18: 'a photo of a person running',
    19: 'a photo of a person working on',
    20: 'a photo of a person skiing',
    21: 'a photo of a person surfing',
    22: 'a photo of a person skateboarding',
    23: 'a photo of a person smiling',
    24: 'a photo of a person drinking',
    25: 'a photo of a person kicking',
    26: 'a photo of a person pointing',
    27: 'a photo of a person reading',
    28: 'a photo of a person snowboarding',
}


def calculate_probability(verb, obj):
    response = requests.get('http://api.conceptnet.io/relatedness?node1=/c/en/' + verb + '&node2=/c/en/' + obj)
    obj = response.json()
    return obj.get('value')

# objects = ["bell_pepper", "bowl" , "bread" ,"bread_container" ,"cabinet" ,"cheese" ,"cheese_container" ,"condiment_container" ,"cooking_utensil" ,"cucumber" ,"cup" ,"cutting_board" ,"drawer" ,"eating_utensil" ,"egg" ,"fridge" ,"fridge_drawer","grocery_bag","lettuce","oil_container" ,"onion" ,"pan" ,"paper_towel" ,"plate" ,"pot","seasoning_container" ,"sponge" ,"tomato" ,"tomato_container"]

# verbs = ["close", "cut", "divide/pull apart", "mix", "move around", "open", "operate", "pour", "put", "take", "wash"]

# change this as required
dataset = 'vcoco'



if dataset == 'hico':
    verb_file = hico_verb_text_label
    obj_file = hico_obj_text_label
elif dataset == 'vcoco':
    verb_file = vcoco_verb_text_label
    obj_file = vcoco_obj_text_label_our    

# be careful of "a photo of nothing" while tokenizing here
num_verbs=len(verb_file)
num_objects=len(obj_file) # exclude a photo of nothing
objects=[]
verbs = []
# verbs=[
#     'adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with',
#     'buy', 'carry', 'catch', 'chase', 'check', 'clean', 'control', 'cook',
#     'cut', 'cut_with', 'direct', 'drag', 'dribble', 'drink_with', 'drive',
#     'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly',
#     'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 'hug',
#     'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss', 'lasso', 'launch',
#     'lick', 'lie_on', 'lift', 'light', 'load', 'lose', 'make', 'milk', 'move',
#     'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay',
#     'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race',
#     'read', 'release', 'repair', 'ride', 'row', 'run', 'sail', 'scratch',
#     'serve', 'set', 'shear', 'sign', 'sip', 'sit_at', 'sit_on', 'slide',
#     'smell', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 'stick',
#     'stir', 'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 'teach',
#     'text_on', 'throw', 'tie', 'toast', 'train', 'turn', 'type_on', 'walk',
#     'wash', 'watch', 'wave', 'wear', 'wield', 'zip'
# ]

separator='_'
for i in range(num_verbs):
    curr_verb=verb_file[i].split()
    verb_name=curr_verb[5:]
    verb_name=separator.join(verb_name)
    verbs.append(verb_name)

for i in range(num_objects-1):
    # not considering BG
    curr_obj=obj_file[i][1].split()
    object_name=curr_obj[4:]
    object_name=separator.join(object_name)
    objects.append(object_name)

similarity_matrix=torch.zeros(num_objects-1,num_verbs)

for j in range(0, len(objects)):
    print(f"filling for object number {j}")
    for i in range(0, len(verbs)):
        verb = verbs[i]
        obj = objects[j]
        # action = verb + ' ' + obj
        prob = calculate_probability(verb, obj)
        similarity_matrix[j][i]=prob
        # print("Probability for action relating verb ->", verb, "and object ->", obj, "is", prob )
        
torch.save(similarity_matrix, 'similarityMatrix_verb_ing.pt')

# probabilities=[]
# for i in range(0, len(verbs)):
#     print(i)
#     verb = verbs[i]
#     obj = "banana"
#     prob = calculate_probability(verb, obj)
#     probabilities.append(prob)    
#     # print("Probability for action relating verb ->", verb, "and object ->", obj, "is", prob )
# probabilities=torch.Tensor(probabilities)
# res,indices=probabilities.topk(5,largest=True)
# topkverbs=[verbs[i] for i in indices]
# print(indices)
# print(topkverbs)

# print(objects)
# for i in range(num_verbs):
#     curr_verb=hico_verb_text_label[i].split()
#     verb_name=curr_verb[5:]
#     verb_name=separator.join(verb_name)
#     verbs.append(verb_name)
