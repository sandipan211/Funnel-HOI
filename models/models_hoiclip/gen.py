import copy
from typing import Optional, List
import math

import torch
import clip
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as cp
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_verb_text_label, hico_unseen_index
from datasets.vcoco_text_label import *

import ALIP.src.open_alip.my_alip_load as alip_load
from ALIP.src.open_alip import tokenize as alip_tokenize

import SLIP.my_slip_load as slip_load 
from SLIP.tokenizer import SimpleTokenizer as SLIP_Tok

import MetaCLIP.src.mini_clip.my_metaclip_load as metaclip_load
from MetaCLIP.src.mini_clip.factory import get_tokenizer as metaclip_tokenize


def get_all_lo(clip_model, dataset, my_tokenizer, unsqueeze_tok_output):

    if dataset == 'hico':
        obj_file = hico_obj_text_label
    elif dataset == 'vcoco':
        obj_file = vcoco_obj_text_label_our

    # expected output shape: total_num_objects x clip_dim 
    total_num_objects=len(obj_file)
    objects=[obj_file[i][1] for i in range(total_num_objects)]

    if unsqueeze_tok_output:
        text_inputs=torch.cat([my_tokenizer(o).unsqueeze(0) for o in objects]).to("cuda")
    else:
        text_inputs=torch.cat([my_tokenizer(o) for o in objects]).to("cuda")

    clip_text_embeddings=clip_model.encode_text(text_inputs).to("cuda")  # need to add line to check if clip model on cuda or not

    clip_text_embeddings=clip_text_embeddings.float()
    clip_text_embeddings=clip_text_embeddings[:total_num_objects-1,:]
    return clip_text_embeddings.detach()

def get_all_lv(clip_model, dataset, my_tokenizer, unsqueeze_tok_output):

    if dataset == 'hico':
        verb_file = hico_verb_text_label
    elif dataset == 'vcoco':
        verb_file = vcoco_verb_text_label

    total_num_verbs=len(verb_file)
    verbs=[verb_file[i] for i in range(total_num_verbs)]
    
    if unsqueeze_tok_output:
        text_inputs=torch.cat([my_tokenizer(o).unsqueeze(0) for o in verbs]).to("cuda")
    else:
        text_inputs=torch.cat([my_tokenizer(o) for o in verbs]).to("cuda")

    clip_text_embeddings=clip_model.encode_text(text_inputs).to("cuda")  # need to add line to check if clip model on cuda or not
    clip_text_embeddings=clip_text_embeddings.float()
   
    return clip_text_embeddings.detach()

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)

class sumModule(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.sumdim=dim
    def forward(self,l):
        return torch.sum(l,dim=self.sumdim)
    
class GEN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_inter_dec_layrs=3,
                 return_intermediate_dec=False, num_queries=64, clip_dim=768,clip_embed_dim=512, enable_cp=False,zero_shot_type='default',num_cross_attention_layers=3,relatedness_matrix=None, 
                 concat_v_fo=True,gated_cross_attn=False,onlyFo=False,bothFoFv=False,ko=5,kv=5,dataset='hico',efficiency_report=False,b_size=2, vlm_version='ViT-B/32', vlm='clip',vlm_path='',
                 noisy_eval=False, noise_alpha=0.1):
        super().__init__()
        self.dataset = dataset

        self.efficiency_report = efficiency_report
        self.batch_size = b_size
        self.vlm = vlm
        self.vlm_version= vlm_version
        self.vlm_path = vlm_path
        self.unsqueeze_tok_output = False
        if self.vlm == 'clip':
            self.original_clip_model,_=clip.load(self.vlm_version)
            self.my_tokenizer = clip.tokenize
        elif self.vlm == 'alip':
            # print('Using ALIP as VLM model in GEN')
            self.original_clip_model,_=alip_load.load(self.vlm_version, self.vlm_path)
            self.my_tokenizer = alip_tokenize
        elif self.vlm == 'slip':
            # print('Using SLIP as VLM model in GEN')
            self.original_clip_model,_=slip_load.load(self.vlm_version, self.vlm_path)
            self.my_tokenizer = SLIP_Tok()
            self.unsqueeze_tok_output = True
        elif self.vlm == 'metaclip':
            # print('Using MetaCLIP as VLM model')
            self.original_clip_model, _, _ = metaclip_load.load(model_name='ViT-B-32-worldwide@WorldWideCLIP', model_weight=self.vlm_path)
            self.my_tokenizer = metaclip_tokenize()

        self.memory_proj =  nn.Linear((d_model), clip_dim)               # 256 to clip_dim(768)
        self.fv_proj =  nn.Linear((d_model), clip_dim)                   # 256 to clip_dim(768)
        self.lo = get_all_lo(self.original_clip_model, self.dataset, self.my_tokenizer, self.unsqueeze_tok_output)
        self.relatedness = relatedness_matrix
        self.lv = get_all_lv(self.original_clip_model, self.dataset, self.my_tokenizer, self.unsqueeze_tok_output)
        
        
        self.co_attention_o = CoAttention(d_model)
        self.co_attention_v = CoAttention(d_model)
        self.m = nn.AdaptiveAvgPool2d((1,d_model))
        self.m2 = nn.AdaptiveAvgPool2d((1,d_model))
        self.sumModule1 = sumModule(2)
        self.conv1 = nn.Conv1d(256, 128, 1, stride=1)
        self.conv2 = nn.Conv1d(256, 128, 1, stride=1)
        self.concat_v_fo = concat_v_fo
        self.onlyFo = onlyFo
        self.bothFoFv = bothFoFv
        self.ko = ko
        self.kv = kv
        self.noisy_eval = noisy_eval; self.noise_amount = noise_alpha
        if self.noisy_eval:
            print('Perturbing noise into object features during evaluation!!!')
        
        # encoder part
        encoder_layer = TransformerEncoderLayer(d_model, nhead,self.original_clip_model, dim_feedforward,
                                                dropout, activation, normalize_before, enable_cp, clip_dim=clip_embed_dim)
        encoder_norm = LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # encoder part ends

        instance_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation, normalize_before, False)
        instance_decoder_norm = LayerNorm(d_model)
        self.instance_decoder = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)

        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before, False)
        interaction_decoder_norm = LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer,
                                                      num_dec_layers,
                                                      interaction_decoder_norm,
                                                      return_intermediate=return_intermediate_dec)

        clip_interaction_decoder_layer = TransformerDecoderFusionLayer(clip_dim, nhead, dim_feedforward,
                                                                       dropout, activation, normalize_before, enable_cp)
        clip_interaction_decoder_norm = LayerNorm(clip_dim)
        self.clip_interaction_decoder = TransformerDecoderCLIP(clip_interaction_decoder_layer,
                                                               num_inter_dec_layrs,
                                                               clip_interaction_decoder_norm,
                                                               return_intermediate=return_intermediate_dec)
        self.inter_guided_embedd = nn.Embedding(num_queries, clip_dim)
        self.queries2spacial_proj = nn.Linear(d_model, clip_dim)
        self.queries2spacial_proj_norm = LayerNorm(clip_dim)

        self.obj_class_fc = nn.Linear(d_model, clip_dim)
        self.obj_class_ln = LayerNorm(clip_dim)

        self.hoi_cls = None

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.uniform_(self.inter_guided_embedd.weight)

    def perturb_object_probed_features(self, f_o, alpha):
        """
        f_o: object co-attention output features shape (B, hw, C), LayerNormed
        alpha: relative noise strength (e.g., 0.05, 0.1, 0.2)
        """

        B, N, D = f_o.shape

        # For LayerNormed features: E[||f_o||_2] ≈ sqrt(D)
        feature_scale = math.sqrt(D)

        # Isotropic Gaussian noise
        noise = torch.randn_like(f_o)

        # Scale noise magnitude
        epsilon = alpha * feature_scale
        noise = noise * epsilon

        # Add perturbation
        f_o_tilde = f_o + noise

        return f_o_tilde

        
    def getTopKlo(self,p):

        if self.dataset == 'hico':
            obj_file = hico_obj_text_label
        elif self.dataset == 'vcoco':
            obj_file = vcoco_obj_text_label_our
        else:
            print('Dataset not supported')

        # expected output shape : bs x K+1 x (cx2)
        k = self.ko
        batch_size=p.shape[0]
        cx2=p.shape[1]
        similarity = torch.bmm(self.lo.unsqueeze(0).repeat(batch_size,1,1),p.unsqueeze(2)).squeeze()    # shape: bs x total_num

        topklo=torch.zeros(batch_size,k+1,cx2)
        topk_obj_inds = torch.zeros(batch_size, k).long()
        obj_sim_scores = torch.zeros(batch_size, k)
        
        for i in range(batch_size):
            currSimilarity = similarity[i,:]
            res, indices= currSimilarity.topk(k+1,largest=True)
            indices=indices.tolist()
            res=res.tolist()
            if(0 in indices):
                human_index=indices.index(0)
                indices.remove(0)
                res.pop(human_index)
            else:
                indices.pop(self.ko)
                res.pop(self.ko)  

            labelNames=[obj_file[index][1] for index in indices]
            topk_obj_inds[i] = torch.Tensor(indices).long()
            obj_sim_scores[i] = torch.Tensor(res).long()
    
            topklo[i,0,:]=self.lo[0,:]
            for j in range(1,k+1):
                currindex=indices[j-1]
                topklo[i,j,:]=self.lo[currindex,:]   
        return topklo, topk_obj_inds, obj_sim_scores

    def getTopKlv(self, obj_sim_matrix, relatedness_matrix, topk_obj_inds, p, k_cNet=10):
    
        if self.dataset == 'hico':
            verb_file = hico_verb_text_label
        elif self.dataset == 'vcoco':
            verb_file = vcoco_verb_text_label
        else:
            print('Dataset not supported')
        
        batch_size=p.shape[0]
        cx2=p.shape[1]
        k = self.kv


        #assuming we have a 81x117 correlation matrix from ConceptNet
        relatedness = torch.argsort(relatedness_matrix, dim=1, descending=True)
        relatedness = relatedness[topk_obj_inds, :k_cNet] #k_cNet is the number of verbs per object acquired from relatedness matrix
        # shape = (bs, num_objs, num_verbs) 
        
        observed_lv = self.lv[relatedness]      
        # shape of observed_lv should be (8, 5, 10, 512)
        similarity = torch.einsum('bcij,bjk->bcik', observed_lv, p.unsqueeze(2)).squeeze(3)    # shape = (8, 5, 10) 
        obj_weights = torch.exp(obj_sim_matrix.unsqueeze(2)).repeat(1, 1, k_cNet)

        obj_weights = obj_weights.to(similarity.device)
        similarity = similarity * obj_weights # shape = (8, 5, 10)

        topklv=torch.zeros(batch_size,k,cx2)
        topk_verb_indices = torch.zeros(batch_size, k).long()

        for b in range(batch_size):
            # method 1: obtain object-wise top-verb 
            # top_per_obj_ind = torch.argsort(similarity[b, :], dim=1, descending=True)[:,0] # shape = (5, 1)
            # currVerbs = relatedness[b, :]
            # currVerbs = currVerbs[range(len(currVerbs)), top_per_obj_ind].tolist() # shape = (5, 1) => top verb per obj
            # top_verb_per_object = [verb_file[index] for index in currVerbs] 
            # batch_objs = [obj_file[index][1] for index in topk_obj_inds[b]]
            # assuming only 5 verbs, whether overlapping or not   
            # for i in range(len(batch_objs)):
            #     print(f'Obj: {batch_objs[i]} -> topmost verb: {top_verb_per_object[i]}')


            # method 2: without looking at verbs object-wise
            sim, rel = similarity[b, :].view(-1), relatedness[b, :].view(-1)                                                                        
            top_verb_indices = torch.argsort(sim, descending=True)
   
            topk_verbs = []
            j = 0
            while len(topk_verbs) < k:
                verb_ind = rel[top_verb_indices[j]].item()
                if verb_ind not in topk_verbs:
                    topk_verb_indices[b][len(topk_verbs)]=verb_ind
                    topk_verbs.append(verb_file[verb_ind])

                    topklv[b,j]=self.lv[verb_ind]
                        
                j += 1
        # taking method 2 
        return topklv, topk_verb_indices
    
    def forward(self, src, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed, clip_model, clip_proj,
                clip_src, targets):
        # the clip_model here is the MODIFIED CLIP (with contextual features as extra output)
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)                   # src: hw x bs x c (in our case bs=2(in testing),8(in training),c=256,hw varies)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)       # pos_embed: hw x bs x c
        num_queries = query_embed_h.shape[0]            

        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)

        mask = mask.flatten(1)                                  # mask: bs x hw
        ins_tgt = torch.zeros_like(ins_query_embed)
        
        # encoder part
        # original_clip_model is the NON-MODIFIED one
        if self.efficiency_report:
            print('Replacing original CLIP features by junk in gen.py')
            original_clip_visual = torch.randn((self.batch_size, 512)).to(mask.device)
            original_clip_visual = original_clip_visual.float()
        else:
            original_clip_visual = self.original_clip_model.encode_image(clip_src)
            original_clip_visual = original_clip_visual.float()
               memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, clip_model=clip_model,original_clip_model=self.original_clip_model, targets=targets, clip_visual=original_clip_visual)
                                                      # shape: bsx512
        if(self.onlyFo or self.bothFoFv):
            topLo, topk_obj_inds, obj_sim_scores =self.getTopKlo(original_clip_visual)
            fo = self.co_attention_o(src,mask,original_clip_visual,topLo)
            fo = torch.permute(fo,(0,3,2,1))      # shape: bsx(hw)xtotal_numxc
            fo = self.m(fo)                       # shape: bsx(hw)x1xc
            fo = torch.squeeze(fo)                # shape: bsx(hw)xc

        if self.noisy_eval:
            fo = self.perturb_object_probed_features(fo, alpha=self.noise_amount)
        
        if(self.bothFoFv):
            topLv, topk_verb_indices = self.getTopKlv(obj_sim_scores,self.relatedness,topk_obj_inds,original_clip_visual)              # shape: bs x total_num_top_verbs x (cx2))
 
            if self.concat_v_fo:
                v_proj = self.conv1(src.permute(1,2,0))
                fo_proj = self.conv2(fo.permute(0,2,1))                                                                                   # shape: bs x 128 x hw
                fv = self.co_attention_v((torch.cat((v_proj,fo_proj),1)).permute(2,0,1),mask,original_clip_visual,topLv)

            else:
                # add V and fo along the channel dimension (256)
                fv = self.co_attention_v(torch.add(src.permute(1,2,0), fo.permute(0,2,1)).permute(2,0,1),mask,original_clip_visual,topLv)
        
            fv = torch.permute(fv,(0,3,2,1))      # shape: bsx(hw)xtotal_numxc
            fv = self.m2(fv)                      # shape: bsx(hw)x1xc
            fv = torch.squeeze(fv)                # shape: bsx(hw)xc
 
        if(self.bothFoFv):
            memory =  memory + fv.permute(1,0,2)
        elif(self.onlyFo):
            memory =  memory + fo.permute(1,0,2)
            
        inst_decoder_input = memory
        ins_hs = self.instance_decoder(ins_tgt, inst_decoder_input, memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)

        
        ins_hs = ins_hs.transpose(1, 2)
        h_hs = ins_hs[:, :, :num_queries, :]
        o_hs = ins_hs[:, :, num_queries:, :]

        memory = self.obj_class_ln(self.obj_class_fc(memory))

        inter_hs = (h_hs + o_hs) / 2.0
        inter_hs = self.queries2spacial_proj(inter_hs[-1]) # taking from last decoder layer
        inter_hs = self.queries2spacial_proj_norm(inter_hs)
        dtype = inter_hs.dtype

        if self.efficiency_report:
            print('Replacing modifiedCLIP features by junk in gen.py')
            clip_cls_feature = torch.randn((self.batch_size, 512)).to(mask.device)
            clip_visual = torch.randn((self.batch_size, 50, 768)).to(mask.device)
        else:
            clip_cls_feature, clip_visual = clip_model.encode_image(clip_src)

        clip_cls_feature = clip_cls_feature / clip_cls_feature.norm(dim=1, keepdim=True)
        clip_cls_feature = clip_cls_feature.to(dtype)
        clip_visual = clip_visual.to(dtype)

        # print(f'clip visual embedding shape:  {clip_visual.shape}')
        # print(f'clip cls token shape:  {clip_cls_feature.shape}')



        with torch.no_grad():
            clip_hoi_score = clip_cls_feature @ self.hoi_cls.T
            clip_hoi_score = clip_hoi_score.unsqueeze(1)

        clip_cls_feature = clip_cls_feature.unsqueeze(1).repeat(1, num_queries, 1)
        inter_hs = self.clip_interaction_decoder(inter_hs.permute(1, 0, 2),
                                                 clip_visual.permute(1, 0, 2), sup_memory=memory)

        inter_hs = inter_hs @ clip_proj.to(dtype)
        inter_hs = inter_hs.permute(0, 2, 1, 3)

        return h_hs, o_hs, inter_hs, clip_cls_feature, clip_hoi_score, clip_visual @ clip_proj.to(dtype)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, clip_model,original_clip_model,targets,clip_visual,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos,clip_model=clip_model,original_clip_model=original_clip_model, targets=targets, clip_visual=clip_visual)

        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerDecoderCLIP(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, sup_memory=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(output.shape) == 4:
                output = output[i]
            else:
                # only this branch will be used, we only use last human/object query and pass one layer decoder block
                output = output
            output = layer(output, memory, sup_memory=sup_memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class CoAttention(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model=d_model
        self.c1=64                                  # arbitrary choice for c1
        self.c2=64                                  # arbitrary choice for c2
        self.wv=nn.Linear((d_model),self.c1)           # expects [*,Hin] -> [*,Hout]
        self.wl=nn.Linear((d_model),self.c1)           # expects [*,Hin] -> [*,Hout]
        self.wv1=nn.Linear((d_model),self.c2)          # expects [*,Hin] -> [*,Hout]
        # due to below hardcoded line works only for 512-dim features now (256 x 2 = 512)
        self.wl1=nn.Linear(2,self.c2)                # expects [*,Hin] -> [*,Hout]
        self.wl2=nn.Linear(self.c2,(d_model))          # expects [*,Hin] -> [*,Hout]
        self.wv2=nn.Linear(self.c2,(d_model))          # expects [*,Hin] -> [*,Hout]
        self.wv3=nn.Linear((d_model),self.c2)          # expects [*,Hin] -> [*,Hout]
        self.wl3=nn.Linear(2,self.c2)                # expects [*,Hin] -> [*,Hout]
        self.normFinal = LayerNorm((d_model))


    def forward(self,v,src_key_padding_mask,p,l):
        # expected output shape: bsxcxtotal_num_objectsx(hw)
        v = v.permute(1,2,0)
        l=l.to('cuda')

        batch_size=l.shape[0]
        total_num=l.shape[1]
        # due to below hardcoded line works only for 512-dim features now (256 x 2 = 512)
        channels=(l.shape[2]//2)
        hw=v.shape[2]
        fo = torch.zeros(v.shape[0],total_num,v.shape[1],v.shape[2])
        l = l.reshape(l.shape[0],l.shape[1],(l.shape[2]//2),2) 
        p = p.reshape(p.shape[0],(p.shape[1]//2),2) 
        
        v2=self.wv(torch.transpose(v, 1, 2))      # shape: bs x (hw) x c1
        l2=self.wl(torch.transpose(l, 2, 3))      # shape: bs x total_num_objects x 2 x c1
        v2=torch.transpose(v2, 1, 2)              # shape: bs x c1 x (hw)
        l2=torch.transpose(l2, 2, 3)              # shape: bs x total_num_objects x c1 x 2

        # Compute the affinity matrix
        affinity_matrix = torch.bmm(torch.transpose(v2, 1, 2), torch.transpose(l2,1,2).view(batch_size,self.c1,-1))  # Affinity between v2 and l2 shape: (bsx(hw)xc1 * bsxc1x(total_num_objectsx2)) = bsx(hw)x(total_numx2))
        affinity_matrix = affinity_matrix.view(batch_size,hw,total_num,2)
        affinity_matrix = torch.transpose(affinity_matrix,1,2)                      # shape: bsxtotal_numx(hw)x2
        
        # assymmetric coattention visual part
        t1v = torch.bmm(torch.transpose(v, 1, 2),p)                          # shape: ((bsxhwxc * bsxcx2)=bsxhwx2) 
        t1v = t1v.unsqueeze(1).repeat(1,total_num,1,1)                       # shape: bsxtotal_numx(hw)x2
        t1v = t1v + affinity_matrix                                          # shape: bsxtotal_numx(hw)x2
        t1v = self.wl1(t1v)                                                  # shape: bsxtotal_numx(hw)xc2
        
        t2v = self.wv1(torch.transpose(l, 2, 3))                                                  # shape: bsxtotal_numx2xc2

        t2v = torch.bmm(affinity_matrix.reshape([(batch_size*total_num),hw,2]),t2v.reshape([(batch_size*total_num),2,self.c2]))      # shape:((bsxtotal_num)x(hw)x2)x((bsxtotal_num)x2xc2)))=(bsxtotal_num)x(hw)xc2 
        t2v=t2v.reshape([batch_size,total_num,hw,self.c2])                                                                           # shape: bsxtotal_numx(hw)xc2 
        
        hv = torch.tanh(t1v+t2v)                  # shape: bsxtotal_numx(hw)xc2

        av = self.wv2(hv)                         # shape: bsxtotal_numx(hw)xc
        av = F.softmax(av, dim=2)                 # check which dimension to take softmax        
        
        # assymmetric coattention semantic part
        t1ltemp=torch.bmm(torch.transpose(l, 2, 3).reshape([batch_size,(total_num*2),channels]),p)   # shape: (bsx(total_numx2)xc * bsxcx2)=bsx(total_numx2)x2
        t1ltemp = t1ltemp.reshape([batch_size,total_num,2,2])
        t1l = torch.bmm(affinity_matrix.reshape([(batch_size*total_num),hw,2]),t1ltemp.reshape([(batch_size*total_num),2,2]))            # shape: ((bsxtotal_num)xhwx2 * (bsxtotal_num)x2x2)=(bsxtotal_num)xhwx2
        t1l = t1l.reshape([batch_size,total_num,hw,2])
        t1l = self.wl3(t1l)                                 # shape: bsxtotal_numx(hw)xc2
        
        t2l = torch.bmm(affinity_matrix.reshape([(batch_size*total_num),hw,2]),(torch.transpose(p, 1, 2).unsqueeze(1).repeat(1,total_num,1,1)).reshape([(batch_size*total_num),2,channels]))            # shape: bsxtotal_numx(hw)x2 *  bs x total_num x 2 x c
        t2l = t2l.reshape([batch_size,total_num,hw,channels])
        t2l = t2l + torch.transpose(v, 1, 2).unsqueeze(1).repeat(1,total_num,1,1)                 # shape: bsxtotal_numx(hw)xc
        
        t2l = self.wv3(t2l)                                 # shape: bsxtotal_numx(hw)xc2
        hl = torch.tanh(t1l+t2l)                            # shape: bsxtotal_numx(hw)xc2

        al = self.wl2(hl)                                   # shape: bsxtotal_numx(hw)xc
        al = F.softmax(al, dim=2)                           # check which dimension to take softmax

        fo=self.normFinal(av+al)                              # shape: bsxtotal_numx(hw)xc
        fo = torch.permute(fo,(0,3,1,2))                      # shape: bsxcxtotal_num_objectsx(hw)
        
        return fo

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead,original_clip_model, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False, clip_dim=512):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.enable_cp:
            def _inner_forward(args):
                src_inner, q_inner, k_inner, src_mask_inner, src_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=src_inner, attn_mask=src_mask_inner,
                                      key_padding_mask=src_key_padding_mask_inner)[0]
                return src_inner

            src2 = cp.checkpoint(_inner_forward, (src, q, k, src_mask, src_key_padding_mask))
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, clip_model,original_clip_model,targets,clip_visual,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
                
        if self.normalize_before:
            curr = self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        else:
            curr = self.forward_post(src, src_mask, src_key_padding_mask, pos)
        return curr

class TransformerDecoderFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, sup_memory=None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):


        q = k = self.with_pos_embed(tgt, query_pos)
        if self.enable_cp:
            def _inner_forward(args):
                tgt_inner, q_inner, k_inner, tgt_mask_inner, tgt_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=tgt_inner, attn_mask=tgt_mask_inner,
                                           key_padding_mask=tgt_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward, (tgt, q, k, tgt_mask, tgt_key_padding_mask))
        else:
            # print(f'For eq. (6), q: {q.shape}, k: {k.shape}, v: {tgt.shape}')
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # print(f'Output shape from self attention: {tgt2.shape}')
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.enable_cp:
            def _inner_forward_co(args):
                tgt_inner, query_pos_inner, memory_inner, pos_inner, memory_mask_inner, memory_key_padding_mask_inner = args
                src_inner = self.multihead_attn(query=self.with_pos_embed(tgt_inner, query_pos_inner),
                                       key=self.with_pos_embed(memory_inner, pos_inner),
                                       value=memory_inner, attn_mask=memory_mask_inner,
                                       key_padding_mask=memory_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward_co, (tgt, query_pos, memory, pos, memory_mask, memory_key_padding_mask))
        else:
            # print(f'For eq. (7), q: {tgt.shape}, k: {memory.shape}, v: {memory.shape}')
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
           
        tgt2 = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt2)

        tgt3 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(sup_memory, pos),
                                   value=sup_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        

        tgt3 = tgt + self.dropout2(tgt3)
        tgt3 = self.norm2(tgt3)
        tgt = tgt2 + tgt3

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, sup_memory=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
     
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, sup_memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.enable_cp:
            def _inner_forward(args):
                tgt_inner, q_inner, k_inner, tgt_mask_inner, tgt_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=tgt_inner, attn_mask=tgt_mask_inner,
                                           key_padding_mask=tgt_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward, (tgt, q, k, tgt_mask, tgt_key_padding_mask))
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.enable_cp:
            def _inner_forward_co(args):
                tgt_inner, query_pos_inner, memory_inner, pos_inner, memory_mask_inner, memory_key_padding_mask_inner = args
                src_inner = self.multihead_attn(query=self.with_pos_embed(tgt_inner, query_pos_inner),
                                       key=self.with_pos_embed(memory_inner, pos_inner),
                                       value=memory_inner, attn_mask=memory_mask_inner,
                                       key_padding_mask=memory_key_padding_mask_inner)[0]
                return src_inner

            tgt2 = cp.checkpoint(_inner_forward_co, (tgt, query_pos, memory, pos, memory_mask, memory_key_padding_mask))
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_gen(args):
    return GEN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_queries=args.num_queries,
        num_inter_dec_layrs=args.inter_dec_layers,
        zero_shot_type=args.zero_shot_type,
        clip_embed_dim=args.clip_embed_dim,
        num_cross_attention_layers=args.num_cross_attention_layers,
        relatedness_matrix=args.relatedness_matrix,
        concat_v_fo = args.concat_v_fo,
        gated_cross_attn = args.gated_cross_attn,
        onlyFo = args.onlyFo,
        bothFoFv = args.bothFoFv,
        ko=args.ko,
        kv=args.kv,
        dataset=args.dataset_file,
        efficiency_report=args.efficiency_report,
        b_size=args.batch_size,
        vlm_version=args.clip_model,
        vlm=args.vlm_model,
        vlm_path=args.vlm_model_path,
        noisy_eval=args.noisy_eval,
        noise_alpha=args.noise_alpha
    )




class SwiGlu(nn.Module):
    def __init__(self, parameter):
        super().__init__()    
        self.beta = parameter

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        # beta parameter not used
        return F.silu(gate) * x

def _get_activation_fn(activation, parameter=None):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "sigmoid":
        return F.sigmoid
    if activation == "SwiGlu":
        return SwiGlu(parameter)
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
