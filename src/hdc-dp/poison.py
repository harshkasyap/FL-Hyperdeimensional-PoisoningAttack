import copy, cv2, enum, heapq, os, sys, torch
from functools import partial
from multiprocessing import Pool, Process
from mxnet import nd as mnd
import numpy as np

import sim

def hdc_dp(hdc_args, flip_labels, poison_percent, hdc_client_data):
    data = hdc_client_data[0]
    data = list(data)
    mal_data = list()

    train_vectors = hdc_client_data[1]
    proj = hdc_client_data[2]
    proj_inv = hdc_client_data[3]

    for source_label, target_label in flip_labels.items():
        b_arr = train_vectors[target_label]
        norm_b = sim.norm(b_arr)
        total_occurences = len([1 for _, label in data if label == source_label])
        poison_count = poison_percent * total_occurences
        if poison_percent == -1:
            poison_count = total_occurences

        label_poisoned = 0
        for index, _ in enumerate(data):
            data[index] = list(data[index])
            
            if data[index][1] == source_label:
                img = data[index][0].reshape(1, hdc_args["one_d_len"])
                img_enc = sim.dot(img.detach().numpy(), proj.detach().numpy())
                img_enc = torch.from_numpy(img_enc)
                
                c_arr = img_enc.reshape(hdc_args["hdc_proj_len"])
                p_arr = copy.deepcopy(c_arr)

                dot_mb = hdc_args["scale_dot"] * sim.dot(b_arr, c_arr)
                norm_c = sim.norm(c_arr)
                norm_m = norm_c
                sim_mg = 1
                
                kwargs = {"scale_norm": hdc_args["scale_norm"]} if "scale_norm" in hdc_args else {}

                for _index in range(3410):
                    p_arr, dot_mb, norm_m, sim_mg, updated = sim.cosine_coord_vector_adapter(b_arr, p_arr, _index, dot_mb, norm_m, sim_mg, c_arr, norm_c, norm_b, **kwargs)

                    if _index > 3400:
                        _p_arr = p_arr.reshape(1, hdc_args["hdc_proj_len"])
                        p_img = sim.dot(_p_arr.detach().numpy(), proj_inv.detach().numpy())
                        p_img = torch.from_numpy(p_img)
                        p_img = p_img.view(hdc_args["view"][0], hdc_args["view"][1], hdc_args["view"][2])
                        #print("Shapes", data[index][0].shape, p_img.shape)
                        print("Simily", sim.cosine_similarity(data[index][0].view(-1), p_img.view(-1)))
                        #if sim.cosine_similarity(data[index][0].view(-1), p_img.view(-1)) > 0.999:
                        mal_data.append(tuple([p_img, target_label]))

                #data[index] = [p_img, target_label]
                label_poisoned += 1

            data[index] = tuple(data[index])

            if label_poisoned >= poison_count:
                break

    if poison_percent == -1:
        return tuple(mal_data)        

    data = data + mal_data
    return tuple(data)