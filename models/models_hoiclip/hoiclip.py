import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
from ModifiedCLIP import clip

# change SS for ACMMM 20204 rebuttal - experimenting with ALIP
import ModifiedALIP.src.open_alip.my_alip_load as alip_load
from ModifiedALIP.src.open_alip import tokenize as alip_tokenize


from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label_our
from datasets.static_hico import HOI_IDX_TO_ACT_IDX

from ..backbone import build_backbone
from ..matcher import build_matcher
from .gen import build_gen

# from myVisualization import myVisualizationAveragePooling,myVisualizationfo,myVisualizationSum,gradCAMAveragePooling,gradCAMfo,gradCAMSum
# hwArray,adaptiveAverageActivations,sumActivations,foActivations=[],[],[],[]
# adaptiveAverageGradients,sumGradients,foGradients=[],[],[]

def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


class HOICLIP(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        self.dataset = transformer.dataset
        hidden_dim = transformer.d_model
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.inter2verb = MLP(args.clip_embed_dim, args.clip_embed_dim // 2, args.clip_embed_dim, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.clip_model, self.preprocess = clip.load(self.args.clip_model)

        # change SS for ACMMM 2024 rebuttal
        if args.vlm_model== 'clip':
            self.clip_model, self.preprocess = clip.load(self.args.clip_model)
            self.my_tokenizer = clip.tokenize
        elif args.vlm_model == 'alip':
            # print('Using ALIP as VLM model in HOICLIP')
            self.clip_model, self.preprocess = alip_load.load(self.args.clip_model, self.args.vlm_model_path)
            self.my_tokenizer = alip_tokenize

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label_our
            unseen_index = None
        ###############################################


        clip_label, obj_clip_label, v_linear_proj_weight, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_CLIP(hoi_text_label, obj_text_label, unseen_index, args.no_clip_cls_init)
        num_obj_classes = len(obj_text) - 1  # del nothing
        self.clip_visual_proj = v_linear_proj_weight

        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim),
        )

        if unseen_index:
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
        else:
            unseen_index_list = []

        if self.args.dataset_file == 'hico':
            verb2hoi_proj = torch.zeros(117, 600)
            select_idx = list(set([i for i in range(600)]) - set(unseen_index_list))
            for idx, v in enumerate(HOI_IDX_TO_ACT_IDX):
                verb2hoi_proj[v][idx] = 1.0
            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj[:, select_idx], requires_grad=False)
            self.verb2hoi_proj_eval = nn.Parameter(verb2hoi_proj, requires_grad=False)

            self.verb_projection = nn.Linear(args.clip_embed_dim, 117, bias=False)
            self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            self.verb_weight = args.verb_weight
        else:
            verb2hoi_proj = torch.zeros(29, 263)
            for i in vcoco_hoi_text_label.keys():
                verb2hoi_proj[i[0]][i[1]] = 1

            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj, requires_grad=False)
            self.verb_projection = nn.Linear(args.clip_embed_dim, 29, bias=False)
            self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            self.verb_weight = args.verb_weight
            # print(f'weight shape: {self.verb_projection.weight.shape}')

        if args.with_clip_label:
            if args.fix_clip_label:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text), bias=False)
                self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
                for i in self.visual_projection.parameters():
                    i.require_grads = False
            else:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text))
                self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)

            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600, bias=False)
                self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        else:
            self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))

        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            if args.fix_clip_label:
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1, bias=False)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
                for i in self.obj_visual_projection.parameters():
                    i.require_grads = False
            else:
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        else:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.transformer.hoi_cls = clip_label / clip_label.norm(dim=-1, keepdim=True)

        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def init_classifier_with_CLIP(self, hoi_text_label, obj_text_label, unseen_index, no_clip_cls_init=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # take class names of all HOIs and get their semantics
        # text_inputs = torch.cat([clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])

        # change SS for ACMMM 2024 rebuttal
        text_inputs = torch.cat([self.my_tokenizer(hoi_text_label[id]) for id in hoi_text_label.keys()])

        # in text_inputs_del, get the semantics for seen HOIs
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        # text_inputs_del = torch.cat(
        #     [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        # obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])

        # change SS for ACMMM 2024 rebuttal
        text_inputs_del = torch.cat(
            [self.my_tokenizer(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        obj_text_inputs = torch.cat([self.my_tokenizer(obj_text[1]) for obj_text in obj_text_label])

        clip_model = self.clip_model
        clip_model.to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_inputs.to(device))
            text_embedding_del = clip_model.encode_text(text_inputs_del.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            v_linear_proj_weight = clip_model.visual.proj.detach()

            # change SS
            # print(f'v_linear_proj: {v_linear_proj_weight.shape}')

        if not no_clip_cls_init:
            print('\nuse clip text encoder to init classifier weight\n')
            return text_embedding.float(), obj_text_embedding.float(), v_linear_proj_weight.float(), \
                   hoi_text_label_del, obj_text_inputs, text_embedding_del.float()
        else:
            print('\nnot use clip text encoder to init classifier weight\n')
            return torch.randn_like(text_embedding.float()), torch.randn_like(
                obj_text_embedding.float()), torch.randn_like(v_linear_proj_weight.float()), \
                   hoi_text_label_del, obj_text_inputs, torch.randn_like(text_embedding_del.float())

    def forward(self, samples, is_training=True, clip_input=None, targets=None):
        if not isinstance(samples, NestedTensor) and not self.args.efficiency_report:
            # change SS - flow of control doesn't reach here, so samples are already of NestedTensor type
            samples = nested_tensor_from_tensor_list(samples)
        
        # print("from hoiclip.py")
        # print(f"samples shape: {samples.shape}")
        # print(f"targets shape: {len(targets)}")
        # print(f"targets: {(targets)}")
        # for target in targets:
        #     print(f"name: {target['filename']}")
        # print(f"targets[0]: {(targets[0])}")
        # print(f"targets[1]: {(targets[1])}")
        # print(f"clip_input shape: {clip_input.shape}")

        # change SS for ACMMM 2024 rebuttal - for flop analysis
        if self.args.efficiency_report:
            src, pos = torch.randn((self.args.batch_size, 2048, 24, 29)), torch.randn((self.args.batch_size, 256, 24, 29))
            mask = torch.randn((self.args.batch_size, 24, 29))
            src = src.to(self.args.device)
            pos_final = pos.to(self.args.device)
            mask = mask.to(self.args.device)

        else:
            features, pos = self.backbone(samples)
            # print(f"features shape: {features[-1].size(0)}")
            # print(f"clip visual proj shape: {self.clip_visual_proj.shape}")
            src, mask = features[-1].decompose()
            assert mask is not None
            pos_final = pos[-1]
        # h_hs, o_hs, inter_hs, clip_cls_feature, clip_hoi_score, clip_visual = self.transformer(self.input_proj(src), mask,
        #                                         self.query_embed_h.weight,
        #                                         self.query_embed_o.weight,
        #                                         self.pos_guided_embedd.weight,
        #                                         pos[-1], self.clip_model, self.clip_visual_proj, clip_input)
        h_hs, o_hs, inter_hs, clip_cls_feature, clip_hoi_score, clip_visual = self.transformer(self.input_proj(src), mask,
                                                self.query_embed_h.weight,
                                                self.query_embed_o.weight,
                                                self.pos_guided_embedd.weight,
                                                pos_final, self.clip_model, self.clip_visual_proj, clip_input, targets)

        # print(f"{adaptiveAverageActivations[0].shape} {adaptiveAverageGradients[0].shape}")
        # print(f"{sumActivations[0].shape} {sumGradients[0].shape}")
        # print(f"{foActivations[0].shape} {foGradients[0].shape}")
        # myVisualizationAveragePooling(adaptiveAverageActivations[0],hwArray[0][0],hwArray[0][1])
        # myVisualizationSum(sumActivations[0],hwArray[0][0],hwArray[0][1])
        # myVisualizationfo(foActivations[0],hwArray[0][0],hwArray[0][1])
        # exit(0)

        # logging above variables
        # print(h_hs.shape)
        # print(o_hs.shape)
        # print(inter_hs.shape)
        # print(clip_cls_feature.shape)
        # print(clip_hoi_score.shape)
        # print(clip_visual.shape)
        # done
        # for making tsne plot
        # interaction_feature_tsne = torch.mean(inter_hs[-1],dim=1)
        # print(targets)
        # print(inter_hs[-1].shape)
        # exit(0)

        outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()

        if self.args.with_obj_clip_label:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(o_hs)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
        else:
            outputs_obj_class = self.obj_class_embed(o_hs)

        if self.args.with_clip_label:
            logit_scale = self.logit_scale.exp()
            # inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone()
            # print(f"outputs_inter_hs : {outputs_inter_hs.shape}")
            verb_hs = self.inter2verb(inter_hs)                                       # O_verb
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)                 # use this
            verb_hs = verb_hs / verb_hs.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' \
                    and (self.args.eval or not is_training):
                # logging ointer to npy file for prompt learning
                # print(outputs_inter_hs[-1].shape)
                # done
                
                # print(f"inter_hs : {inter_hs.shape}")
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
                # print(f"outputs_hoi_class : {outputs_hoi_class.shape}")
                outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj_eval
                # print(f"outputs_verb_class : {outputs_verb_class.shape}")
                outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
                # print(f"outputs_hoi_class : {outputs_hoi_class.shape}")
            else:
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
                outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj
                outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
        else:
            inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone()
            outputs_hoi_class = self.hoi_class_embedding(inter_hs)
        # print(outputs_verb_class.shape)
        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 'clip_visual': clip_visual,
               'clip_cls_feature': clip_cls_feature, 'hoi_feature': inter_hs[-1], 'clip_logits': clip_hoi_score}
        # check dimension of above 
        if self.args.with_mimic:
            out['inter_memory'] = outputs_inter_hs[-1]
        if self.aux_loss:
            if self.args.with_mimic:
                aux_mimic = outputs_inter_hs
            else:
                aux_mimic = None

            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_class,
                                                            outputs_sub_coord, outputs_obj_coord,
                                                            aux_mimic)

        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord, outputs_inter_hs=None):

        if outputs_hoi_class.shape[0] == 1:
            outputs_hoi_class = outputs_hoi_class.repeat(self.dec_layers, 1, 1, 1)
        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1],
                       }
        if outputs_inter_hs is not None:
            aux_outputs['inter_memory'] = outputs_inter_hs[-self.dec_layers: -1]
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.args = args
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # change SS for ACMMM 2024 rebuttal
        if self.args.with_mimic:
            if self.args.vlm_model == 'clip':
                self.clip_model, _ = clip.load(self.args.clip_model, device=device)
            elif self.args.vlm_model =='alip':
                self.clip_model, _ = alip_load.load(self.args.vlm_model, self.args.vlm_model_path)
        else:
            self.clip_model = None
        self.alpha = self.args.alpha

        # change SS: added the following members
        self.rho_eps = self.args.rho_eps
        self.hc_eps = self.args.hc_eps
        self.kappa = self.args.kappa
        self.obj_induced_interaction_relatedness = None # shape = [80, num_seen_inertactions]
        self.proposed_loss = self.args.proposed_loss

        # for loss ablations
        self.ordis_factors = self.args.ordis_factors

    # change SS: added cost argument in all the loss definitions


    def loss_obj_labels(self, outputs, targets, indices, num_interactions, opt_query_wise_hun_cost, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o


        # change SS
        # print(target_classes.shape)
        # print(target_classes_o) # GT object labels for corresponding to every query match in every image
        # print(f'obj preds [1][10]:\n {src_logits[1][10]}')
        # print(f'obj preds [1][16]:\n {src_logits[1][16]}')
        # src_scores = src_logits.sigmoid()
        # obj_labels = F.softmax(src_scores, -1)[..., :-1].max(-1)[1]
        # obj_labels = F.softmax(src_scores, -1)[0][0]
        # print(f'obj labels predicted {obj_labels.shape}: {obj_labels}') # shape = (81,)
        # query_wise_target_objs = target_classes[idx]
        # print(f'query_wise_target_objs: {query_wise_target_objs}')
        # exit(0)



        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # print(loss_obj_ce.shape)
        # exit(0)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions, opt_query_wise_hun_cost):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions, opt_query_wise_hun_cost):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, opt_query_wise_hun_cost, topk=5):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']
        dtype = src_logits.dtype
        # print('print seen lists')
        # print(self.args.seen_hoi2obj)
        # print(self.args.seen_hoi2verb)

        idx = self._get_src_permutation_idx(indices)

        # change SS: obtaining object label info for the matched queries
        obj_label_info = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])

        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)]).to(dtype)
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        src_logits = _sigmoid(src_logits)

        # change SS
        # print(f'permutation (list1 = batch, list2 = optimal query slots) {idx}')
        
        # print(target_classes.shape)
        # print(target_classes_o.shape) #[total number of h/o pairs in the batch, number of training hoi classes]
        # print(target_classes_o)
        # 8 is the query slot which came out for first image
        # print(target_classes[0][8])
        
        # print(f'interaction vector [0][8]:\n {target_classes[0][8]}\n')
        # print(f'interaction vector [1][16]:\n {target_classes[1][16]}\n')

        # print(src_logits[1][20])  
        # for i in [10, 16]:      
        #     pos_inds = target_classes[1][i].eq(1)
        #     print(pos_inds)
        #     print(f'pred sigmoid value of GT action(s): {src_logits[1][i][pos_inds]}')
        #     print(f'max sigmoid pred at action {torch.argmax(src_logits[1][i])}')


        # change SS: our proposed loss instead of usual focal loss
        # loss_hoi_ce = self._obj_induced_focal_loss(src_logits, target_classes, opt_query_wise_hun_cost, obj_label_info, idx, weights=None, alpha=self.alpha, rho_eps=self.rho_eps, hc_eps=self.hc_eps, kappa=self.kappa)
        # losses = {'loss_hoi_labels': loss_hoi_ce}

        if self.proposed_loss:
            # print("My loss")
            loss_hoi_ce = self._obj_induced_focal_loss(src_logits, target_classes, opt_query_wise_hun_cost, obj_label_info, idx, weights=None, alpha=self.alpha, rho_eps=self.rho_eps, hc_eps=self.hc_eps, kappa=self.kappa)
        else:
            loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        
        losses = {'loss_hoi_labels': loss_hoi_ce}


        _, pred = src_logits[idx].topk(topk, 1, True, True)
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error'] = torch.from_numpy(np.array(
            rel_labels_error)).to(src_logits.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions, opt_query_wise_hun_cost):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def mimic_loss(self, outputs, targets, indices, num_interactions, opt_query_wise_hun_cost):
        src_feats = outputs['inter_memory']
        src_feats = torch.mean(src_feats, dim=1)

        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
        loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
        losses = {'loss_feat_mimic': loss_feat_mimic}
        return losses
    def reconstruction_loss(self, outputs, targets, indices, num_interactions, opt_query_wise_hun_cost):
        raw_feature = outputs['clip_cls_feature']
        hoi_feature = outputs['hoi_feature']

        loss_rec = F.l1_loss(raw_feature, hoi_feature)
        return {'loss_rec': loss_rec}

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        # change SS
        # print(f'pos loss [1][10]=\n {pos_loss[1][10]}\n')
        # print(f'neg loss [1][16]=\n {neg_loss[1][10]}\n')
        # print(f'pos loss [1][16]=\n {pos_loss[1][16]}\n')
        # print(f'neg loss [1][16]=\n {neg_loss[1][16]}\n')


        # print(f'neg loss = {neg_loss[0][8]}')


        num_pos = pos_inds.float().sum() # total number of GT interactions in [bs, num_queries, num_seen_inters]
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            # the case where a query slot doesn't contain any object 
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        #print(f"loss device: {loss.device}")
        return loss


    # change SS

    def formulate_pa(self, betas, zetas, rhos, rho_eps):

        if self.ordis_factors == 'default':
            return torch.sigmoid(betas * torch.div(zetas, (rhos + rho_eps + 2.0)))
        elif self.ordis_factors == 'beta':
            return torch.sigmoid(betas)
        elif self.ordis_factors == 'zeta':
            return torch.sigmoid(zetas)
        elif self.ordis_factors == 'delta':
            return torch.sigmoid(torch.div(torch.ones(zetas.shape).to(zetas.device), (rhos + rho_eps + 2.0)))
        elif self.ordis_factors == 'beta_delta':
            return torch.sigmoid(torch.div(betas, (rhos + rho_eps + 2.0)))
        elif self.ordis_factors == 'beta_zeta':
            return torch.sigmoid(betas * zetas)
        elif self.ordis_factors == 'zeta_delta':
            return torch.sigmoid(torch.div(zetas, (rhos + rho_eps + 2.0)))


    def _obj_induced_focal_loss(self, pred, gt, opt_query_wise_hun_cost, obj_label_info, perm, weights=None, alpha=0.25, rho_eps=1e-14, hc_eps=1e-7, kappa=-2):

        '''
        
        pred: the predictions from interaction decoder (Q_i = [bs, num_queries, num_seen_interactions])
        gt  : the target interactions (T_i = [bs, num_queries, num_seen_interactions])
        cost_matrix: the cost matrix computed during Hungarian matching (C_i = [bs, num_queries, num_ho_pairs_in_batch])
        opt_query_wise_hun_cost: (HC_i = [num_ho_pairs_in_batch])
        obj_label_info: target_classes_o as returned from loss_obj_labels(). An array of size [num_ho_pairs_in_batch] that contains matched-query-wise object ground truths.
        perm: the optimal permutations obtained from Hungarian matching
        '''
        # print(f"pred: {pred.device}")
        # print(f"gt: {gt.device}")
        # print(f"opt_query_wise_hun_cost: {opt_query_wise_hun_cost.device}")
        # print(f"obj_label_info: {obj_label_info.device}")
        # print(f" perm: {perm[0].device}")
        
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # compute confidence magnitude (beta)
        #print(perm)
        #print(opt_query_wise_hun_cost)
        bs, num_queries, num_classes = gt.shape
        betas = torch.zeros(bs, num_queries)
        betas[perm] = torch.log(1 + torch.pow(torch.add(opt_query_wise_hun_cost, hc_eps), -kappa)) # working index wise but check computed value
        betas = betas.unsqueeze(2).repeat(1, 1, num_classes)
        
        betas = betas.to(gt.device)
        
        # print('check valid betas') => verified correctness!
        # print(f'betas shape = {betas.shape}')
        # print(betas[0][6])
        # print(betas[4][44])
        # print(betas[4][59])
        # print(betas[6][39])
        # print(betas[0][18])
        
        # compute object-induced scaling factor (rho)
        rhos = torch.zeros(gt.shape)
        # print(self.obj_induced_interaction_relatedness)
        # print(self.obj_induced_interaction_relatedness.sum())
        # print(self.obj_induced_interaction_relatedness.shape)


        rhos[perm] = (self.obj_induced_interaction_relatedness[obj_label_info])
        # rhos[perm] = (self.obj_induced_interaction_relatedness.detach()[obj_label_info])

        # rhos[perm] = (int(obj_label_info.sum()))
        rhos = rhos.to(gt.device)
        # print(rhos)
        # exit(0)
        # rhos => verify later


        # print(pos_inds[perm].shape)   # shape = [num_ho_pairs_in_batch, num_interactions]
        # print(pos_inds[perm])
 
        # compute discrepancy scores of each interaction
        query_wise_num_pos = pos_inds[perm].float().sum(axis=1) 
        # [num_ho_pairs_in_batch] array: ele = entry[i] denotes we have "ele" number of GT interactions for ith matched query slot
        zetas = torch.zeros(gt.shape)
        gt_scores = pred[perm] * pos_inds[perm] # shape = [num_ho_pairs_in_batch, num_interactions] 
        # print(f'gt_scores shape = {gt_scores.shape}')
        # print(gt_scores.sum(axis=1))
        # print(f'query wise GT counts: \n{query_wise_num_pos}')

        avg_gt_scores = torch.div(gt_scores.sum(axis=1), query_wise_num_pos).unsqueeze(1).repeat(1, num_classes)
        # print(avg_gt_scores)
        discrepancy = (pred[perm] - avg_gt_scores) * neg_inds[perm] # keeping non-zero discrepancy only for the non-GT preds
       
        zetas = zetas.to(discrepancy.device)
        # print(f"zetas device: {zetas.device}")
        # print(f"discrepancy device: {discrepancy.device}")
        zetas[perm] = discrepancy
        # print(zetas)
        # exit(0)

        # compute interactive penalty actuator
        # the actuator produces the loss terms in a controlled way according to the object-induced interactiveness, when prediction signals are supplied by the model
        # print(f"betas:{betas.device}")
        # print(f"rhos:{rhos.device}")
        
        penalty_actuator = self.formulate_pa(betas, zetas, rhos, rho_eps) # [bs, num_queries, num_seen_interactions]
        # print(f"penalty_actuator:{penalty_actuator.device}")
        # print(f"pos_inds:{pos_inds.device}")
        loss = 0

        # remove this
        # penalty_actuator = torch.sigmoid(betas * torch.div(zetas, ( rho_eps + 2.0))) # [bs, num_queries, num_seen_interactions]

        pos_loss = penalty_actuator * alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]
        

        neg_loss = penalty_actuator * (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        # change SS
        # print(f'pos loss [1][10]=\n {pos_loss[1][10]}\n')
        # print(f'neg loss [1][16]=\n {neg_loss[1][10]}\n')
        # print(f'pos loss [1][16]=\n {pos_loss[1][16]}\n')
        # print(f'neg loss [1][16]=\n {neg_loss[1][16]}\n')
        # print(f'perm: {perm}')
        # print(f'q costs: {opt_query_wise_hun_cost}')
        # print(neg_loss.shape)
        # print(pos_loss.shape)
        # print(f'neg loss = {neg_loss[0][8]}')


        num_pos = pos_inds.float().sum() # total number of GT interactions in [bs, num_queries, num_seen_inters]
        # rowsum_pos, rowsum_neg = torch.sum(pos_loss, 1), torch.sum(neg_loss, 1)
        # print(f'pos loss shape: {torch.div(torch.sum(rowsum_pos, 1), num_pos).shape}')
        # print(f'neg loss shape: {torch.div(torch.sum(rowsum_neg, 1), num_pos).shape}')
        # print(f'pos loss: {torch.div(torch.sum(rowsum_pos, 1), num_pos)}')
        # print(f'neg loss: {torch.div(torch.sum(rowsum_neg, 1), num_pos)}')

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()


        if num_pos == 0:
            # the case where a query slot doesn't contain any object 
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        # print(f"my loss: {loss}")  
        # exit(0)
        return loss






    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # change SS: added cost argument
    def get_loss(self, loss, outputs, targets, indices, num, opt_query_wise_hun_cost, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
                'feats_mimic': self.mimic_loss,
                'rec_loss': self.reconstruction_loss
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, opt_query_wise_hun_cost, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets

        # change SS: modified outputs from matcher 
        indices, opt_query_wise_hun_cost = self.matcher(outputs_without_aux, targets)
        #change SS
        # print(f'Hungarian algo output shape: {len(indices)}')
        # print(indices)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # change SS: added cost argument
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions, opt_query_wise_hun_cost))

        # exit(0)
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            # print('Using auxiliary losses') => being used indeed
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, opt_query_wise_hun_cost = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss =='rec_loss':
                        continue
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, opt_query_wise_hun_cost, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']
        clip_visual = outputs['clip_visual']
        clip_logits = outputs['clip_logits']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'), 'clip_visual': clip_visual[index].to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:], 'clip_logits': clip_logits[index].to('cpu')})

        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_gen(args)

    model = HOICLIP(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    if args.with_clip_label:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.with_mimic:
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef

    if args.with_rec_loss:
        weight_dict['loss_rec'] = args.rec_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes']
    if args.with_mimic:
        losses.append('feats_mimic')

    if args.with_rec_loss:
        losses.append('rec_loss')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors
