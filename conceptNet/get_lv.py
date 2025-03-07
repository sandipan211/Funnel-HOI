def getTopKlv(self, obj_sim_matrix, relatedness_matrix, topk_obj_inds, p, k=5, k_cNet=10):
    # expected output shape : bs x K+1 x (cx2)
    batch_size=p.shape[0]
    cx2=p.shape[1]


    #assuming we have a 81x117 correlation matrix from ConceptNet
    relatedness = torch.argsort(relatedness_matrix, dim=1, descending=True)
    relatedness = relatedness[topk_obj_inds, :k_cNet] #k_cNet is the number of verbs per object acquired from ConceptNet relatedness matrix
    # shape = (bs, num_objs, num_verbs) 
    # observed_lv = relatedness.view(-1)
    observed_lv = self.lv[relatedness]
    # shape of observed_lv should be (8, 5, 10, 512)
    similarity = torch.einsum('bcij,bjk->bcik', observed_lv, p.unsqueeze(2)).squeeze()    # shape = (8, 5, 10) 

    # obj_sim_matrix shape = (bs, num_objs) = (8, 5)
    obj_weights = torch.exp(obj_sim_matrix).repeat(1, 1, k_cNet)
    similarity = similarity * obj_weights # shape = (8, 5, 10)

    for b in range(batch_size):
        # method 1: obtain object-wise top-verb 
        top_per_obj_ind = torch.argsort(similarity[b, :], dim=1, descending=True)[:,0] # shape = (5, 1)
        currVerbs = relatedness[b, :]
        currVerbs = currVerbs[range(len(currVerbs)), top_per_obj_ind].tolist() # shape = (5, 1) => top verb per obj
        top_verb_per_object = [hico_verb_text_label[index] for index in currVerbs] 
        batch_objs = [hico_obj_text_label[index][1] for index in topk_obj_inds[b]]
        # assuming only 5 verbs, whether overlapping or not   
        for i in range(len(batch_objs)):
            print(f'Obj: {batch_objs[i]} -> topmost verb: {top_verb_per_object[i]}')


        # method 2: without looking at verbs object-wise
        similarity, relatedness = similarity[b, :].view(-1), relatedness[b, :].view(-1)
        top_verb_indices = torch.argsort(similarity, dim=1, descending=True)
        topk_verbs = []
        j = 0
        while len(topk_verbs) < k:
            verb_ind = relatedness[top_verb_indices[j]]
            if verb_ind not in topk_verbs:
                topk_verbs.append(hico_verb_text_label[verb_ind])
                print(f'Verb acquired: {hico_verb_text_label[verb_ind]}')
            j += 1


    exit(0)



 