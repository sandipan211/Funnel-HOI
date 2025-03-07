import copy
from typing import Optional, List

import torch
import clip
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as cp
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_verb_text_label, hico_unseen_index
from datasets.vcoco_text_label import *

# change SS for ACMMM 20204 rebuttal - experimenting with ALIP
import ALIP.src.open_alip.my_alip_load as alip_load
from ALIP.src.open_alip import tokenize as alip_tokenize

# mdl,_=clip.load('ViT-B/32')



def get_all_lo(clip_model, dataset, my_tokenizer):

    if dataset == 'hico':
        obj_file = hico_obj_text_label
    elif dataset == 'vcoco':
        obj_file = vcoco_obj_text_label_our

    # expected output shape: total_num_objects x clip_dim 
    total_num_objects=len(obj_file)
    objects=[obj_file[i][1] for i in range(total_num_objects)]
    # print(objects)
    # text_inputs=torch.cat([clip.tokenize(o) for o in objects]).to("cuda")
    # change SS for ACMMM 2024 rebuttal
    text_inputs=torch.cat([my_tokenizer(o) for o in objects]).to("cuda")

    clip_text_embeddings=clip_model.encode_text(text_inputs).to("cuda")  # need to add line to check if clip model on cuda or not
    # clip_text_embeddings=clip_text_embeddings.to("cpu")
    # print(f"clip embedding shape: {clip_text_embeddings.shape}")
    # clip_text_embeddings=clip_text_embeddings.unsqueeze(0).repeat(batch_size,1,1)
    clip_text_embeddings=clip_text_embeddings.float()
    # print(clip_text_embeddings)
    # trying with just one output class
    clip_text_embeddings=clip_text_embeddings[:total_num_objects-1,:]
    # print(f"this size should be 80 {clip_text_embeddings.shape}")
    return clip_text_embeddings.detach()

def get_all_lv(clip_model, dataset, my_tokenizer):

    if dataset == 'hico':
        verb_file = hico_verb_text_label
    elif dataset == 'vcoco':
        verb_file = vcoco_verb_text_label

    # expected output shape: total_num_verbs x clip_dim 
    total_num_verbs=len(verb_file)
    verbs=[verb_file[i] for i in range(total_num_verbs)]
    # print(len(verbs))
    # text_inputs=torch.cat([clip.tokenize(o) for o in verbs]).to("cuda")
    # change SS for ACMMM 2024 rebuttal
    text_inputs=torch.cat([my_tokenizer(o) for o in verbs]).to("cuda")

    clip_text_embeddings=clip_model.encode_text(text_inputs).to("cuda")  # need to add line to check if clip model on cuda or not
    # clip_text_embeddings=clip_text_embeddings.to("cpu")
    # print(clip_text_embeddings)
    clip_text_embeddings=clip_text_embeddings.float()
    # print(clip_text_embeddings)
    # print(f"this size should be 117x512 {clip_text_embeddings.shape}")
    # exit(0)
    return clip_text_embeddings.detach()

# def get_all_lo_rho(clip_model):
#     # expected output shape: total_num_objects x clip_dim 
#     total_num_objects=len(obj_file)
#     objects=[obj_file[i][1] for i in range(total_num_objects)]
#     objects=[objects[i][4:] for i in len(objects)]
#     # print(objects)
#     text_inputs=torch.cat([clip.tokenize(o) for o in objects]).to("cuda")
#     clip_text_embeddings=clip_model.encode_text(text_inputs).to("cuda")  # need to add line to check if clip model on cuda or not
#     # clip_text_embeddings=clip_text_embeddings.to("cpu")
#     # print(f"clip embedding shape: {clip_text_embeddings.shape}")
#     # clip_text_embeddings=clip_text_embeddings.unsqueeze(0).repeat(batch_size,1,1)
#     clip_text_embeddings=clip_text_embeddings.float()
#     # print(clip_text_embeddings)
#     # trying with just one output class
#     clip_text_embeddings=clip_text_embeddings[:total_num_objects-1,:]
#     # print(f"this size should be 80 {clip_text_embeddings.shape}")
#     return clip_text_embeddings.detach()

# def get_all_lv_rho(clip_model):
#     # expected output shape: total_num_verbs x clip_dim 
#     total_num_verbs=len(verb_file)
#     verbs=[verb_file[i] for i in range(total_num_verbs)]
#     verbs=[verbs[i][5:] for i in range(len(verbs))]
#     # print(len(verbs))
#     text_inputs=torch.cat([clip.tokenize(o) for o in verbs]).to("cuda")
#     clip_text_embeddings=clip_model.encode_text(text_inputs).to("cuda")  # need to add line to check if clip model on cuda or not
#     # clip_text_embeddings=clip_text_embeddings.to("cpu")
#     # print(clip_text_embeddings)
#     clip_text_embeddings=clip_text_embeddings.float()
#     # print(clip_text_embeddings)
#     # print(f"this size should be 117x512 {clip_text_embeddings.shape}")
#     # exit(0)
#     return clip_text_embeddings.detach()



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
                 concat_v_fo=True,gated_cross_attn=False,onlyFo=False,bothFoFv=False,ko=5,kv=5,dataset='hico',efficiency_report=False,b_size=2, vlm_version='ViT-B/32', vlm='clip',vlm_path=''):
        super().__init__()
        # mine
        self.dataset = dataset

        # change SS ACMMM 2024 rebuttal
        self.efficiency_report = efficiency_report
        self.batch_size = b_size
        self.vlm = vlm
        self.vlm_version= vlm_version
        self.vlm_path = vlm_path
        if self.vlm == 'clip':
            self.original_clip_model,_=clip.load(self.vlm_version)
            self.my_tokenizer = clip.tokenize
        elif self.vlm == 'alip':
            # print('Using ALIP as VLM model in GEN')
            self.original_clip_model,_=alip_load.load(self.vlm_version, self.vlm_path)
            self.my_tokenizer = alip_tokenize



        # print(f"dmodel: {d_model}")                                       # 256
        self.memory_proj =  nn.Linear((d_model), clip_dim)               # 256 to clip_dim(768)
        self.fv_proj =  nn.Linear((d_model), clip_dim)                   # 256 to clip_dim(768)
        # change SS for ACMMM 2024 rebuttal
        self.lo = get_all_lo(self.original_clip_model, self.dataset, self.my_tokenizer)
        self.relatedness = relatedness_matrix
        # self.relatedness[57]=-float('inf')
        self.lv = get_all_lv(self.original_clip_model, self.dataset, self.my_tokenizer)
        
        
        self.co_attention_o = CoAttention(d_model)
        self.co_attention_v = CoAttention(d_model)
        self.m = nn.AdaptiveAvgPool2d((1,d_model))
        self.m2 = nn.AdaptiveAvgPool2d((1,d_model))
        self.sumModule1 = sumModule(2)
        self.conv1 = nn.Conv1d(256, 128, 1, stride=1)
        self.conv2 = nn.Conv1d(256, 128, 1, stride=1)
        # cross_attention_layer=CrossAttentionLayer(d_model,nhead,dropout,clip_dim)
        # self.gated_cross_attention=CrossAttention(cross_attention_layer,num_cross_attention_layers)
        # self.p_proj = nn.Linear(clip_dim, (d_model))
        # self.fi_proj =  nn.Linear(clip_dim, (d_model))                   # clip_dim(768) to 256
        self.concat_v_fo = concat_v_fo
        self.onlyFo = onlyFo
        self.bothFoFv = bothFoFv
        self.ko = ko
        self.kv = kv
        # self.gated_cross_attn = gated_cross_attn

        # mine ends
        
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
        
    def getTopKlo(self,p):

        # change SS for ACMMM 2024 rebuttal - find obj_file (if any) - those are changes
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
        # print(f"shapes: lo({self.lo.shape}), p({p.shape})")
        # this is written considering lo(total_num x (cx2)), p(bs x (cx2)) where p is clip visual encoder
        similarity = torch.bmm(self.lo.unsqueeze(0).repeat(batch_size,1,1),p.unsqueeze(2)).squeeze()    # shape: bs x total_num
        # print(f"shape should be: bsxtotal_num: {similarity.shape}")
        # extracting array of shape bs to find rank of human among 80
        # print(similarity)
        # human_rank=torch.argsort(similarity, dim=1)[:,0] + 1
        # print(f"human ranks: {human_rank}")
        topklo=torch.zeros(batch_size,k+1,cx2)
        topk_obj_inds = torch.zeros(batch_size, k).long()
        obj_sim_scores = torch.zeros(batch_size, k)
        
        for i in range(batch_size):
            # now we want to extract topk(human always included, so in that sense, think of this as k+1)
            currSimilarity = similarity[i,:]
            res, indices= currSimilarity.topk(k+1,largest=True)
            # print(res)
            indices=indices.tolist()
            res=res.tolist()
            # print(indices)
            if(0 in indices):
                human_index=indices.index(0)
                indices.remove(0)
                res.pop(human_index)
            else:
                # indices.pop(5)
                # res.pop(5)

                # change SS: removing hard-coding
                indices.pop(self.ko)
                res.pop(self.ko)  

            labelNames=[obj_file[index][1] for index in indices]
            topk_obj_inds[i] = torch.Tensor(indices).long()
            obj_sim_scores[i] = torch.Tensor(res).long()
            # print(labelNames)
            # print(indices)
            topklo[i,0,:]=self.lo[0,:]
            for j in range(1,k+1):
                currindex=indices[j-1]
                topklo[i,j,:]=self.lo[currindex,:]   
        # print(topklo.shape)
        return topklo, topk_obj_inds, obj_sim_scores

    def getTopKlv(self, obj_sim_matrix, relatedness_matrix, topk_obj_inds, p, k_cNet=10):
    
        # change SS for ACMMM 2024 rebuttal - find verb_file - those are changes
        if self.dataset == 'hico':
            verb_file = hico_verb_text_label
        elif self.dataset == 'vcoco':
            verb_file = vcoco_verb_text_label
        else:
            print('Dataset not supported')
        
        # expected output shape : bs x K+1 x (cx2)
        batch_size=p.shape[0]
        cx2=p.shape[1]
        k = self.kv


        #assuming we have a 81x117 correlation matrix from ConceptNet
        relatedness = torch.argsort(relatedness_matrix, dim=1, descending=True)
        relatedness = relatedness[topk_obj_inds, :k_cNet] #k_cNet is the number of verbs per object acquired from relatedness matrix
        # shape = (bs, num_objs, num_verbs) 
        
        observed_lv = self.lv[relatedness]
        # print(f'observed lv: {observed_lv.shape}')
        # print(f'p : {p.unsqueeze(2).shape}') # shape = (8, 512, 1)
        
        # shape of observed_lv should be (8, 5, 10, 512)
        similarity = torch.einsum('bcij,bjk->bcik', observed_lv, p.unsqueeze(2)).squeeze(3)    # shape = (8, 5, 10) 
        # last dimension (3) has to be squeezed specifically, otherwise only writing squeeze() gives us error when ko = 1 - so similarity = (8, 1, 10, 1) and squeeze() will erroneously reduce both the 1-dimensions
        # print(similarity.shape)

        # obj_sim_matrix shape = (bs, num_objs) = (8, 5)
        obj_weights = torch.exp(obj_sim_matrix.unsqueeze(2)).repeat(1, 1, k_cNet)
        
        # print(f"obj_weights shape:{obj_weights.shape}")
        # print(f"similarity shape:{similarity.shape}")
        obj_weights = obj_weights.to(similarity.device)
        similarity = similarity * obj_weights # shape = (8, 5, 10)

        # change AT
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
            # print(f'rel/sim shape before view: {relatedness.shape}, {similarity.shape}')
            sim, rel = similarity[b, :].view(-1), relatedness[b, :].view(-1)
            # print(f'rel/sim shape after view: {rel.shape}, {sim.shape}')
                                                                        
            top_verb_indices = torch.argsort(sim, descending=True)
            # print(f'top verb inds shape: {top_verb_indices.shape}')
            # print(top_verb_indices)
            topk_verbs = []
            j = 0
            while len(topk_verbs) < k:
                verb_ind = rel[top_verb_indices[j]].item()
                if verb_ind not in topk_verbs:
                    topk_verb_indices[b][len(topk_verbs)]=verb_ind
                    topk_verbs.append(verb_file[verb_ind])

                    topklv[b,j]=self.lv[verb_ind]
                        
                    # print(f'Verb acquired: {verb_file[verb_ind]}')
                j += 1
        # change AT: taking method 2 
        return topklv, topk_verb_indices
    
    def forward(self, src, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed, clip_model, clip_proj,
                clip_src, targets):
        # the clip_model here is the MODIFIED CLIP (with contextual features as extra output)
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        # print(f"h: {h},w: {w}")
        src = src.flatten(2).permute(2, 0, 1)                   # src: hw x bs x c (in our case bs=2(in testing),8(in training),c=256,hw varies)
        # print(f"src shape: {src.shape}")
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)       # pos_embed: hw x bs x c
        # print(f"pos_embed shape: {pos_embed.shape}")            
        num_queries = query_embed_h.shape[0]            

        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)

        # print(f'query_embed shape:  {query_embed_h.shape}')
        # print(f'ins_query_embed shape:  {ins_query_embed.shape}')

        mask = mask.flatten(1)                                  # mask: bs x hw
        # print(f"mask shape: {mask.shape}")
        ins_tgt = torch.zeros_like(ins_query_embed)
        
        # encoder part
        
        # print(clip_src.shape)
        # original_clip_visual = self.original_clip_model.encode_image(clip_src)
        # change SS for ACMMM 2024 rebuttal - original_clip_model is the NON-MODIFIED one
        if self.efficiency_report:
            print('Replacing original CLIP features by junk in gen.py')
            original_clip_visual = torch.randn((self.batch_size, 512)).to(mask.device)
            original_clip_visual = original_clip_visual.float()
        else:
            # original_clip_visual = mdl.encode_image(clip_src)
            original_clip_visual = self.original_clip_model.encode_image(clip_src)
            original_clip_visual = original_clip_visual.float()
        # print(clip_src[0,:,:,:])
        # print(original_clip_visual[0,:])
        # exit(0)
        # print("from gen.py forward")
        # print(clip_src)
        # print(original_clip_visual)
        # print(f"clip_src shape: {clip_src.shape}")
        # print(f"original_clip_visual shape: {original_clip_visual.shape}")
        # print(f"datatype of clip_visual({(clip_visual.dtype)}), clip_proj({(clip_proj.dtype)})")
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, clip_model=clip_model,original_clip_model=self.original_clip_model, targets=targets, clip_visual=original_clip_visual)
        # print(memory.shape)                                                                                     # shape: hwxbsxc
        # encoder part ends
        # my part
        # memory_768 = self.memory_proj(memory.permute(1,0,2))                                                    # shape: bsxhwxc
        # # print(memory_768.shape)                                                                                   
        # memory_768 = memory_768.permute(1,0,2)                                                                  # shape: hwxbsxc
        # print(memory_768.shape)                                                                                   
        # check if clip_visual is right p
        # print(f"p_512: {original_clip_visual.shape}")                                                         # shape: bsx512
        if(self.onlyFo or self.bothFoFv):
            topLo, topk_obj_inds, obj_sim_scores =self.getTopKlo(original_clip_visual)
            fo = self.co_attention_o(src,mask,original_clip_visual,topLo)
            fo = torch.permute(fo,(0,3,2,1))      # shape: bsx(hw)xtotal_numxc
            fo = self.m(fo)                       # shape: bsx(hw)x1xc
            fo = torch.squeeze(fo)                # shape: bsx(hw)xc
        
        if(self.bothFoFv):
            topLv, topk_verb_indices = self.getTopKlv(obj_sim_scores,self.relatedness,topk_obj_inds,original_clip_visual)              # shape: bs x total_num_top_verbs x (cx2))
            # print(topLv.shape)

            # to calculate the accuracy of calculation of toplo,toplv
            # first we would like to get a list of all different objects and verbs present in annotations
            # for i in range(len(targets)):
            #     matched_o=0
            #     matched_v=0
            #     interactive_images=0
            #     targets[i]["numMatches_o"]=0          
            #     targets[i]["numMatches_v"]=0          
            #     targets[i]["numinteractive_images"]=0 
            #     target=targets[i]
            #     hois = target['hois']
            #     verblist=[]
            #     no_interaction = True
            #     for hoi in hois:
            #         if(hoi[2]!=57):
            #             no_interaction=False
            #         verblist.append(hoi[2])
            #     if(no_interaction):
            #         continue
            #     interactive_images+=1
            #     for index in topk_verb_indices[i]:
            #         if(index in verblist):
            #             matched_v+=1
            #             break
            #     labels = target['labels']
            #     for index in topk_obj_inds[i]:
            #         if(index in labels):
            #             matched_o+=1
            #             break
            #     targets[i]["numMatches_o"]=matched_o          
            #     targets[i]["numMatches_v"]=matched_v          
            #     targets[i]["numinteractive_images"]=interactive_images 

            if self.concat_v_fo:
                # project V and fo to 128 and concat to get v for fv
                v_proj = self.conv1(src.permute(1,2,0))
                fo_proj = self.conv2(fo.permute(0,2,1))
                # print(v_proj.shape)                                                                                     # shape: bs x 128 x hw
                # print(fo_proj.shape)                                                                                    # shape: bs x 128 x hw
                fv = self.co_attention_v((torch.cat((v_proj,fo_proj),1)).permute(2,0,1),mask,original_clip_visual,topLv)

            else:
                # add V and fo along the channel dimension (256)
                fv = self.co_attention_v(torch.add(src.permute(1,2,0), fo.permute(0,2,1)).permute(2,0,1),mask,original_clip_visual,topLv)
        
            fv = torch.permute(fv,(0,3,2,1))      # shape: bsx(hw)xtotal_numxc
            fv = self.m2(fv)                      # shape: bsx(hw)x1xc
            fv = torch.squeeze(fv)                # shape: bsx(hw)xc
            
            
            # fv_768 = self.fv_proj(fv)
            # if(self.gated_cross_attn):
                # have to comment line to project memory to clip_embed_dim i.e. memory = self.obj_class_ln(self.obj_class_fc(memory))
                # _ , p_768 = clip_model.encode_image(clip_src)                                                               
                # fi_768 = self.gated_cross_attention(fv_768.permute(0,2,1), mask, memory_768,p_768, mask)
                # # print(fi_768.shape)                                                                                       # shape: hwxbsx768
                # fi_256 = self.fi_proj(fi_768)
                # # print(fi_256.shape)                                                                                       # shape: hwxbsx256
                # memory = fi_768
                # inst_decoder_input = fi_256

                # print(p_768.shape)

                # trying setting 2.4 from google doc
                # memory =  memory + fv.permute(1,0,2)
                # p_256 = self.p_proj(p_768.float())
                # fi_256 = self.gated_cross_attention(fv.permute(0,2,1), mask, memory, p_256, mask)
                # inst_decoder_input = memory = fi_256
            # else:
                # print(memory_768.shape)
                # print(fv_768.shape)

                # using projection to clip_embed_dim
                # memory =  memory_768 + fv_768.permute(1,0,2)
                # inst_decoder_input = self.fi_proj(memory)

                # keeping resnet dim
                # have to uncomment line to project memory to clip_embed_dim i.e. memory = self.obj_class_ln(self.obj_class_fc(memory))
            
        if(self.bothFoFv):
            memory =  memory + fv.permute(1,0,2)
        elif(self.onlyFo):
            memory =  memory + fo.permute(1,0,2)
            
        inst_decoder_input = memory

        
        # my part ends
        # print(f'encoder output shape:  {memory.shape}')

        
        # sending 256 projected fi to instance decoder
        # ins_hs = self.instance_decoder(ins_tgt, memory, memory_key_padding_mask=mask,
        #                                pos=pos_embed, query_pos=ins_query_embed)
        ins_hs = self.instance_decoder(ins_tgt, inst_decoder_input, memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)
        
        
        # print(f"instance decoder output: {ins_hs.shape}")
        # print("from gen.py forward")
        # print(f"src shape: {src.shape}")
        # print(f"samples shape: {samples.shape}")
        # print(f"targets shape: {len(targets)}")
        # print(f"targets: {(targets)}")
        # for target in targets:
        #     print(f"name: {target['filename']}")
        
        ins_hs = ins_hs.transpose(1, 2)
        h_hs = ins_hs[:, :, :num_queries, :]
        o_hs = ins_hs[:, :, num_queries:, :]

        # print(f'instance decoder human output shape:  {h_hs.shape}')
        # print(f'instance decoder object output shape:  {o_hs.shape}')



        # original
        # ins_guided_embed = (h_hs + o_hs) / 2.0
        # ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)
        #
        # inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # inter_hs_ori = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
        #                                         pos=pos_embed, query_pos=ins_guided_embed)
        # inter_hs_ori = inter_hs_ori.transpose(1, 2)
        
        memory = self.obj_class_ln(self.obj_class_fc(memory))

        # print(f'encoder output shape after projection to clip dimension:  {memory.shape}')


        # h_hs_detached = h_hs.detach()

        inter_hs = (h_hs + o_hs) / 2.0
        inter_hs = self.queries2spacial_proj(inter_hs[-1]) # taking from last decoder layer
        inter_hs = self.queries2spacial_proj_norm(inter_hs)
        # inter_hs = inter_hs + self.inter_guided_embedd.weight.unsqueeze(0).repeat(bs, 1, 1)

        # print(f'instance decoder output shape after projection to clip dimension:  {inter_hs.shape}')


        dtype = inter_hs.dtype

        # change SS for ACMMM 2024 rebuttal
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
            # obj_score = clip_cls_feature @ self.obj_cls.T
            # obj_hoi_score = obj_score @ self.obj2hoi_proj

            # verb_score = clip_cls_feature @ self.verb_cls.T
            # verb_hoi_score = verb_score @ self.verb2hoi_proj
            # clip_hoi_score += verb_hoi_score * 0.1
            # ignore_idx = clip_hoi_score.sort(descending=True).indices[:, self.topk:]
            # for idx, igx in enumerate(ignore_idx):
            #     clip_hoi_score[idx][igx] *= 0
            clip_hoi_score = clip_hoi_score.unsqueeze(1)

        clip_cls_feature = clip_cls_feature.unsqueeze(1).repeat(1, num_queries, 1)
        
        # print("inter_hs")
        # print(inter_hs.shape)
        # print(inter_hs.permute(1, 0, 2))
        inter_hs = self.clip_interaction_decoder(inter_hs.permute(1, 0, 2),
                                                 clip_visual.permute(1, 0, 2), sup_memory=memory)

        # print(f'interaction output shape after interaction decoder:  {inter_hs.shape}')
        inter_hs = inter_hs @ clip_proj.to(dtype)

        # print(f'interaction output shape after interaction decoder and projection (O_inter):  {inter_hs.shape}')

        inter_hs = inter_hs.permute(0, 2, 1, 3)
        # print(f'interaction output shape after interaction decoder and permute:  {inter_hs.shape}')

        # add
        # ins_guided_embed = (h_hs + o_hs) / 2.0
        # ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)
        # #torch.Size([3, 64, 8, 256])
        #
        # inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # inter_hs = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
        #                                     pos=pos_embed, query_pos=ins_guided_embed)
        # inter_hs = inter_hs.transpose(1, 2)
        return h_hs, o_hs, inter_hs, clip_cls_feature, clip_hoi_score, clip_visual @ clip_proj.to(dtype)

# encoder part
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
        # print(f"output shape: {output.shape}")
        return output
# encoder part ends

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
        # print("memory shape:")
        # print(memory.shape)
        # print("tgt shape:")
        # print(tgt.shape)
        # print(tgt)
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            # print(f"before {output.shape}")
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
            # print(f"after {output.shape}")
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

# my encoder part
class CoAttention(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model=d_model
        self.c1=64                                  # arbitrary choice for c1
        self.c2=64                                  # arbitrary choice for c2
        self.wv=nn.Linear((d_model),self.c1)           # expects [*,Hin] -> [*,Hout]
        self.wl=nn.Linear((d_model),self.c1)           # expects [*,Hin] -> [*,Hout]
        self.wv1=nn.Linear((d_model),self.c2)          # expects [*,Hin] -> [*,Hout]
        # change SS: due to below hardcoded line works only for 512-dim features now (256 x 2 = 512)
        self.wl1=nn.Linear(2,self.c2)                # expects [*,Hin] -> [*,Hout]
        self.wl2=nn.Linear(self.c2,(d_model))          # expects [*,Hin] -> [*,Hout]
        self.wv2=nn.Linear(self.c2,(d_model))          # expects [*,Hin] -> [*,Hout]
        self.wv3=nn.Linear((d_model),self.c2)          # expects [*,Hin] -> [*,Hout]
        self.wl3=nn.Linear(2,self.c2)                # expects [*,Hin] -> [*,Hout]
        # below part moved to gen
        # self.lo = get_all_lo(original_clip_model)
        # self.relatedness = torch.load('../conceptNet/similarityMatrix.pt')
        # self.relatedness[57]=-float('inf')
        # self.lv = get_all_lv(original_clip_model)
        
        self.normFinal = LayerNorm((d_model))

    

    
    def forward(self,v,src_key_padding_mask,p,l):
        # expected output shape: bsxcxtotal_num_objectsx(hw)
        v = v.permute(1,2,0)
        
        # below line commented because coattention unit should be independent of whether it is lo or lv 
        # i.e. it should take generic l as input and this part gets shifted to gen forward
        # l, topk_obj_inds, obj_sim_scores =self.getTopKlo(p)
        
        l=l.to('cuda')
        
        # lv=self.getTopKlv(obj_sim_scores,self.relatedness,topk_obj_inds,p)                      # shape: bs x K^2 x (cx2)
        # print(l.device)
        # this is written considering v(bs x c x hw), l(bs x total_num x (cx2)), p(bs x (cx2)) where p is clip visual encoder
        # print(f"shapes of v({v.shape}), l({l.shape}), p({p.shape})")
        batch_size=l.shape[0]
        total_num=l.shape[1]
        # change SS: due to below hardcoded line works only for 512-dim features now (256 x 2 = 512)
        channels=(l.shape[2]//2)
        hw=v.shape[2]
        # print(f"values: batch_size={batch_size}, total_num={total_num}, channels={channels}, hw={hw}")
        fo = torch.zeros(v.shape[0],total_num,v.shape[1],v.shape[2])
        # print(f"fo shape: {fo.shape}")
        l = l.reshape(l.shape[0],l.shape[1],(l.shape[2]//2),2) 
        p = p.reshape(p.shape[0],(p.shape[1]//2),2) 
        # print(f"shapes of v({v.shape}), l({l.shape}), p({p.shape})")
        
        # now this part is written considering v(bs x c x hw), l(bs x total_num x c x 2), p(bs x c x 2) where p is clip visual encoder 
        # v_t=torch.transpose(v, 1, 2)              # to get into bs x (hw) x c
        # l_t=torch.transpose(l, 2, 3)              # to get into bs x total_num x 2 x c
        # p_t=torch.transpose(p, 1, 2)              # to get into bs x 2 x c
        v2=self.wv(torch.transpose(v, 1, 2))      # shape: bs x (hw) x c1
        l2=self.wl(torch.transpose(l, 2, 3))      # shape: bs x total_num_objects x 2 x c1
        v2=torch.transpose(v2, 1, 2)              # shape: bs x c1 x (hw)
        l2=torch.transpose(l2, 2, 3)              # shape: bs x total_num_objects x c1 x 2

        # now we compute the affinity matrix
        # Transpose v2 for compatibility
        # v2_t = torch.transpose(v2, 1, 2)
        # l3 = torch.transpose(l2,1,2)              # shape: bs x c1 x total_num x 2
        
        # Compute the affinity matrix
        affinity_matrix = torch.bmm(torch.transpose(v2, 1, 2), torch.transpose(l2,1,2).view(batch_size,self.c1,-1))  # Affinity between v2 and l2 shape: (bsx(hw)xc1 * bsxc1x(total_num_objectsx2)) = bsx(hw)x(total_numx2))
        affinity_matrix = affinity_matrix.view(batch_size,hw,total_num,2)
        affinity_matrix = torch.transpose(affinity_matrix,1,2)                      # shape: bsxtotal_numx(hw)x2
        
        # assymmetric coattention visual part
        t1v = torch.bmm(torch.transpose(v, 1, 2),p)                          # shape: ((bsxhwxc * bsxcx2)=bsxhwx2) 
        t1v = t1v.unsqueeze(1).repeat(1,total_num,1,1)                       # shape: bsxtotal_numx(hw)x2
        # print(t1v.shape)
        t1v = t1v + affinity_matrix                                          # shape: bsxtotal_numx(hw)x2
        # print(t1v.shape)
        t1v = self.wl1(t1v)                                                  # shape: bsxtotal_numx(hw)xc2
        
        t2v = self.wv1(torch.transpose(l, 2, 3))                                                  # shape: bsxtotal_numx2xc2
        # print(t2v_t.shape)
        # print(affinity_matrix.shape)
        # print(t2v.shape)
        t2v = torch.bmm(affinity_matrix.reshape([(batch_size*total_num),hw,2]),t2v.reshape([(batch_size*total_num),2,self.c2]))      # shape:((bsxtotal_num)x(hw)x2)x((bsxtotal_num)x2xc2)))=(bsxtotal_num)x(hw)xc2 
        t2v=t2v.reshape([batch_size,total_num,hw,self.c2])                                                                           # shape: bsxtotal_numx(hw)xc2 
        
        hv = torch.tanh(t1v+t2v)                  # shape: bsxtotal_numx(hw)xc2

        av = self.wv2(hv)                         # shape: bsxtotal_numx(hw)xc
        av = F.softmax(av, dim=2)                 # check which dimension to take softmax        
        
        # assymmetric coattention semantic part
        t1ltemp=torch.bmm(torch.transpose(l, 2, 3).reshape([batch_size,(total_num*2),channels]),p)   # shape: (bsx(total_numx2)xc * bsxcx2)=bsx(total_numx2)x2
        t1ltemp = t1ltemp.reshape([batch_size,total_num,2,2])
        # print(t1ltemp.shape)
        t1l = torch.bmm(affinity_matrix.reshape([(batch_size*total_num),hw,2]),t1ltemp.reshape([(batch_size*total_num),2,2]))            # shape: ((bsxtotal_num)xhwx2 * (bsxtotal_num)x2x2)=(bsxtotal_num)xhwx2
        t1l = t1l.reshape([batch_size,total_num,hw,2])
        t1l = self.wl3(t1l)                                 # shape: bsxtotal_numx(hw)xc2
        
        # print(t1l.shape)
        # p_ttemp=torch.transpose(p, 1, 2).unsqueeze(1).repeat(1,total_num,1,1)                                               # shape: bs x total_num x 2 x c
        # t2ltemp = torch.bmm(affinity_matrix.reshape([(batch_size*total_num),hw,2]),(torch.transpose(p, 1, 2).unsqueeze(1).repeat(1,total_num,1,1)).reshape([(batch_size*total_num),2,channels]))            # shape: bsxtotal_numx(hw)x2 *  bs x total_num x 2 x c
        # t2ltemp = t2ltemp.reshape([batch_size,total_num,hw,channels])
        # t2l = torch.transpose(v, 1, 2).unsqueeze(1).repeat(1,total_num,1,1) + t2ltemp                # shape: bsxtotal_numx(hw)xc
        
        t2l = torch.bmm(affinity_matrix.reshape([(batch_size*total_num),hw,2]),(torch.transpose(p, 1, 2).unsqueeze(1).repeat(1,total_num,1,1)).reshape([(batch_size*total_num),2,channels]))            # shape: bsxtotal_numx(hw)x2 *  bs x total_num x 2 x c
        t2l = t2l.reshape([batch_size,total_num,hw,channels])
        t2l = t2l + torch.transpose(v, 1, 2).unsqueeze(1).repeat(1,total_num,1,1)                 # shape: bsxtotal_numx(hw)xc
        
        t2l = self.wv3(t2l)                                 # shape: bsxtotal_numx(hw)xc2
        # print(t2l.shape)
        
        hl = torch.tanh(t1l+t2l)                            # shape: bsxtotal_numx(hw)xc2

        al = self.wl2(hl)                                   # shape: bsxtotal_numx(hw)xc

        # change SS: probably the line below is wrong - changing it after commenting it
        # al = F.softmax(av, dim=2)                           # check which dimension to take softmax
        al = F.softmax(al, dim=2)                           # check which dimension to take softmax

        # add and norm and return fo
        fo=self.normFinal(av+al)                              # shape: bsxtotal_numx(hw)xc
        # fo = torch.transpose(fo, 2, 3)                      # shape: bsxtotal_numxcx(hw)
        fo = torch.permute(fo,(0,3,1,2))                      # shape: bsxcxtotal_num_objectsx(hw)
        
        return fo
        
# class CrossAttention(nn.Module):
#     def __init__(self, cross_attention_layer, num_layers):
#         super().__init__()
#         self.layers = _get_clones(cross_attention_layer, num_layers)
#         self.num_layers = num_layers
    
#     def forward(self,fa,src_key_padding_mask1,memory,p,src_key_padding_mask2):
#         output = memory

#         for i, layer in enumerate(self.layers):
#             output = layer(fa, src_key_padding_mask1, output, p, src_key_padding_mask2)
            
#         return output

# class CrossAttentionLayer(nn.Module):
#     def __init__(self,d_model,nhead,dropout=0.1, gcat_dim=768):
#         super().__init__()
#         self.norm1 = LayerNorm(gcat_dim)         # check which dimensions across which to take norm
#         self.norm2 = LayerNorm(gcat_dim)         # check which dimensions across which to take norm
#         self.self_attn1 = nn.MultiheadAttention(gcat_dim, nhead, dropout=dropout, batch_first=True)
#         self.self_attn2 = nn.MultiheadAttention(gcat_dim, nhead, dropout=dropout, batch_first=True)
#         self.conv1 = nn.Conv1d(gcat_dim, (gcat_dim//2), 1, stride=1)
#         self.conv2 = nn.Conv1d(gcat_dim, (gcat_dim//2), 1, stride=1)
#         # self.activationSwig1 = SwiGlu(1)
#         # self.activationSwig2 = SwiGlu(1)
#         self.activationGlu1 = torch.nn.GLU(dim=1)
#         self.activationGlu2 = torch.nn.GLU(dim=1)
#         self.norm3 = LayerNorm(gcat_dim)
        
#         # self.norm4 = LayerNorm(d_model)
#         self.activation1 = _get_activation_fn('sigmoid')
#         self.activation2 = _get_activation_fn('sigmoid') 

#         # Implementation of Feedforward model
#         dim_feedforward=512
#         self.activation = torch.nn.Sigmoid()
#         self.linear1 = nn.Linear(gcat_dim, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, gcat_dim)


#     def forward(self,fa,src_key_padding_mask1,memory,p,src_key_padding_mask2):
#         # shapes of fa(Bx768xHW), p(Bx768x50) -> clip visual and memory(hwxbsxc) -> vanilla encoder output
#         vanilla_encoder_output = memory.permute(1,2,0)
#         p = p.permute(0,2,1)
#         # print(f"fa: {fa.shape}")                                        # shape: bsx768xhw
#         # print(f"memory: {vanilla_encoder_output.shape}")                # shape: bsx768xhw
#         # print(f"p: {p.shape}")                                          # shape: bsx768x50

#         # not adding the positioning embeddings 
#         # q1 = k2 = v2 = self.with_pos_embed(f1, pos1)
#         # q2 = k1 = v1 = self.with_pos_embed(f2, pos2)
#         v1 = k1 = fa.permute(0,2,1)                                                      # shape: bsxhwx768
#         v2 = k2 = p.permute(0,2,1).float()                                               # shape: bsx50x768
#         q1 = q2 = vanilla_encoder_output.permute(0,2,1)                                  # shape: bsxhwx768

#         f1_att = self.self_attn1(q1, k1, value=v1, attn_mask=None,
#                               key_padding_mask=src_key_padding_mask1)[0] # check which src_key_padding mask needs to be used
#         f2_att = self.self_attn2(q2, k2, value=v2, attn_mask=None,
#                               key_padding_mask=None)[0] # check which src_key_padding mask needs to be used
        
#         f1_att = self.norm1(f1_att + q1)    # shape: BxHWx768 
#         f2_att = self.norm2(f2_att + q2)    # shape: BxHWx768 

#         f1_att = f1_att.permute(0,2,1)
#         f2_att = f2_att.permute(0,2,1)

#         za = self.activationGlu1(f1_att)    # shape: Bx384xHW
#         zi = self.activationGlu2(f2_att)    # shape: Bx384xHW

        
#         # f1_att = self.conv1(f1_att)         # shape: Bx384xHW 
#         # f2_att = self.conv2(f2_att)         # shape: Bx384xHW
#         # # print(f1_att.shape)
#         # # print(f2_att.shape)
#         # # check the swig activation function classes
#         # f1_swig = self.activation1(f1_att)
#         # f2_swig = self.activation2(f2_att)
#         # # print(f1_swig.shape)
#         # # print(f2_swig.shape)
#         # za = f1_att * f1_swig               # shape: Bx384xHW
#         # zi = f2_att * f2_swig               # shape: Bx384xHW
        
#         # print(za.shape)
#         # print(zi.shape)

#         z_final= torch.cat((za,zi),1)       # shape: Bx768xHW
        
#         # print(z_final.shape)
        
#         # check norm here
#         z_final=self.norm3(z_final.permute(0,2,1)) # shape: BxHWx768

#         # print(z_final.shape)
        
#         # now we pass fe through a feed forward network and return
#         fe = z_final                                                                # shape: bsx(hw)xc
#         fe = self.linear2(self.dropout(self.activation(self.linear1(fe))))
#         fe = fe.permute(1,0,2)                                                      # shape: hwxbsxc
#         # print(fe.shape)
#         return fe
    
# my encoder part ends
# encoder part
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

        # mine
        # the below part is shifted to gen forward because we dont want this coattention in every layer
        # self.clip_dim=clip_dim
        # self.co_attention1 = CoAttention(d_model,original_clip_model)
        # self.m = nn.AdaptiveAvgPool2d((1,d_model))
        # self.sumModule1 = sumModule(2)
        # # self.co_attention2 = CoAttention(d_model)
        # # self.co_attention3 = CoAttention(d_model)
        # # self.cross_attention = CrossAttention(d_model,nhead,dropout)
        # mine ends


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # print(f"src before q,k : {src.shape}")
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
        # print(f"src after encoder layer q,k : {src.shape}")
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
        # print("from gen.py encoder")
        # print(f"src shape before q,k : {src.shape}")
        # print(f"src_mask : {src_mask.shape}")
        # print(f"src_key_padding_mask : {src_key_padding_mask.shape}")
        # print(f"pos : {pos.shape}")
        
        # print(f"samples shape: {samples.shape}")
        # print(f"targets shape: {len(targets)}")
        # print(f"targets: {(targets)}")
        # for target in targets:
            # print(f"name: {target['filename']}")

        # mine
        # note that this calculation of lo, la and li is only valid when we are training and we have the targets
        # print("New image")
        # print(targets[0]["filename"])
        # print(targets[0]["labels"])
        # print(targets[0]["obj_labels"])
        # print(obj_file[targets[0]["obj_labels"][0]])
        # print(targets[0]["verb_labels"])
        # print(len(targets[0]["verb_labels"]))
        # print(f'verb index: {(targets[0]["verb_labels"][0])}')
        # print(f'verb index: {torch.argmax(targets[0]["verb_labels"][0])}')
        # # print(targets[0]["hoi_labels"])
        # print(f'hoi index: {torch.argmax(targets[0]["hoi_labels"][0])}')
        

        if self.normalize_before:
            curr = self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        else:
            curr = self.forward_post(src, src_mask, src_key_padding_mask, pos)
        return curr
        # the below part is shifted to gen forward because we dont want this coattention in every layer        
        # fo = self.co_attention1(src,src_key_padding_mask,clip_visual)
        # # src2 = src + fo
        
        # # mine ends
        # ans = fo                                # shape: bsxcxtotal_num_objectsx(hw)
        # # print(fo[0][0])
        # # here shapes: ans(bsxtotal_num_objectsxcx(hw)) , curr((hw)xbsxc)
        # # also expected output: (hw)xbsxc
        # # addition across K dimension(may use later)
        # # addedAns = torch.sum(fo,dim=2)
        # addedAns = self.sumModule1(fo)
        # # print(f"adedAns shape(should be bsxcx(hw)): {addedAns.shape}")
        # # method to take GAP across total_num_objects
        # ans=torch.permute(ans,(0,3,2,1))        # shape: bsx(hw)xtotal_numxc
        # ans = self.m(ans)                       # shape: bsx(hw)x1xc
        # ans = torch.squeeze(ans)                # shape: bsx(hw)xc
        # ans = ans.permute(1,0,2).to(curr.device) + curr
        # ans=ans.to(curr.device)
        # ans = ans + curr
        # return ans

# encoder part ends

class TransformerDecoderFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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
        
        # print(f'enable cp: {self.enable_cp}')
        # print(query_pos.shape) if query_pos is not None else print('No query pos')
        # print(tgt_mask.shape) if tgt_mask is not None else print('No tgt_mask')
        # print(memory_mask.shape) if memory_mask is not None else print('No memory_mask')
        # print(tgt_key_padding_mask.shape) if tgt_key_padding_mask is not None else print('No tgt_key_padding_mask')
        # print(memory_key_padding_mask.shape) if memory_key_padding_mask is not None else print('No memory_key_padding_mask')
        # print(pos.shape) if pos is not None else print('No pos')


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
            # print(f'Output shape from cross attention 1: {tgt2.shape}')
           
        tgt2 = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt2)

        # print(f'For eq. (8), q: {tgt.shape}, k: {sup_memory.shape}, v: {sup_memory.shape}')

        tgt3 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(sup_memory, pos),
                                   value=sup_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        # print(f'Output shape from cross attention 2: {tgt3.shape}')

        # tgt3 = tgt + self.dropout4(tgt3)
        # tgt3 = self.norm4(tgt3)
        tgt3 = tgt + self.dropout2(tgt3)
        tgt3 = self.norm2(tgt3)

        tgt = tgt2 + tgt3

        # print(f'Output shape before feeding to FFN: {tgt.shape}')

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
        # print("from interaction decoder layer")
        # print(f"{memory.shape}")
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
        vlm_path=args.vlm_model_path
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
