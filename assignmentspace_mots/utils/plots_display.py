import numpy as np
from collections import namedtuple
import munkres
import pycocotools.mask as cocomask
import structmots_utils.mots_helper as mots_helper
from cv2 import remap, INTER_NEAREST
from scipy import spatial
import os
import math
import matplotlib.pyplot as plt
import png
import time
import torch
import sys
import networkx as nx
import cv2
import pdb

def show_graph_pyvis(G_actual, name="AssignmentGraph"):
    G = G_actual.copy()

    for k in list(G.nodes):
        if G.nodes[k] == {} or list(G.nodes[k]) == []:
            G.remove_node(k)
    # positions = node_positions(G)

    g = nx.Network()
    # g.from_nx(G)
    for n in list(G.nodes):
        if n != "start" and n != "end":
            g.add_node(n, value=G.nodes[n]["node_cost"])
        else:
            g.add_node(n, value=1)

    for edge in list(G.edges):
        g.add_edge(edge[0], edge[1])
    #
    # for n in g.nodes:
    #   n.update({'physics': False})

    g.show(name+".html")

def plot_all_test(all_vids, vids_y_star, vids_y_GT, path):
    plt.figure()
    clrs = plt.cm.get_cmap('hsv', len(vids_y_star[0]) + 1)  # cm.rainbow(np.linspace(0, 1, len(all_y_star[0])))
    for i in range(0, len(vids_y_star[0])):
        plt.plot([y_st[i] for y_st in vids_y_star], label="y" + str(i) + "_star", color=clrs(i), alpha=0.7)
        plt.plot([y_GT[i] for y_GT in vids_y_GT], linestyle='dashed', color=clrs(i),
                 label="y" + str(i) + "_GT", alpha=0.7)
    # plt.plot(all_iters, [y2_GT for y_st in all_y_star], 'b--', label="y2_GT", alpha=0.7)

    plt.xlabel("vids")
    plt.ylabel("y")
    plt.xticks(list(range(0, 2)), [vid[0] for vid in all_vids])
    plt.savefig(path, pad_inches=0.5)
    # pdb.set_trace()
    plt.close()
    return


def show_masks(prev_masks, warped_prev_masks, current_masks, save_name, caption):
    prev_len = prev_masks.shape[2]
    current_len = current_masks.shape[2]
    ax = []
    fig = plt.figure(figsize=(30, max(prev_len, current_len) * 5))
    for current_i in range(0, current_len):
        current_mask = torch.FloatTensor(current_masks[:, :, current_i])
        for prev_i in range(0, prev_len):
            result_mask = torch.FloatTensor(prev_masks[:, :, prev_i])

            if dir is not None and current_i == 0:
                img = warped_prev_masks[:, :, prev_i]

                ax.append(fig.add_subplot(max(prev_len, current_len), 3, prev_i * 3 + 2))
                ax[-1].set_title("Frame prev: warped object " + str(prev_i))
                plt.axis('off')

                plt.imshow(img)

                img = prev_masks[:, :, prev_i]

                ax.append(fig.add_subplot(max(prev_len, current_len), 3, prev_i * 3 + 1))
                ax[-1].set_title("Frame prev: Object" + str(prev_i))
                plt.axis('off')

                plt.imshow(img)

        if dir is not None:
            img = current_masks[:, :, current_i]
            ax.append(fig.add_subplot(max(prev_len, current_len), 3, (current_i + 1) * 3))
            ax[-1].set_title("Frame t: Object " + str(current_i))
            plt.imshow(img)
            plt.axis('off')
    if dir is not None:
        fig.suptitle(caption)
        plt.savefig(save_name + ".png")
        plt.close()

def node_positions(H, display_graph=False):
    # get node positions as dict
    # positions={n1:[x1,y1],n2=[x2,y2],etc}
    positions = {}
    for i in range(0, len(list(H.nodes))):
        n1 = list(H.nodes)[i]
        idx = int(list(H.nodes)[i].find('_'))

        if (int(n1[:idx]) / 25) % 2 == 0:
            x = int(n1[:idx]) % 25
            y = (int(n1[:idx]) / 25) * 100 + int(n1[idx + 1:])
        else:
            x = 25 - int(n1[:idx]) % 25
            y = (int(n1[:idx]) / 25) * 100 + int(n1[idx + 1:])
        positions[n1] = [x, y]

    ##display

    if display_graph == True:
        nx.draw_networkx(H, positions)
        plt.show()

    return positions

#used in train
def display_data(images, data, y_star, y_gt, G, path="./data.png",test=False ):#,0]):
    boxes=data["boxes"]
    n=len(boxes)
    plt.figure(figsize=(20,20))
    cols=range(0,2)
    rows=range(0,n)
    k=1
    clrs_all = np.array([[0.,0.,0.],[1.,0.,0.], [1.,1.,0.], [0.,1.,0.], [0.,1.,1.], [0.,0.,1.], [1.,0.,1.]])
    clrs_all=np.concatenate([clrs_all, clrs_all * 0.5, 1 - clrs_all * 0.5, (1 - clrs_all) * 0.5,
                             clrs_all*0.25, 1 - clrs_all * 0.25, (1 - clrs_all) * 0.25,
                             clrs_all*0.15, 1 - clrs_all * 0.15, (1 - clrs_all) * 0.15,
                             clrs_all*0.35, 1 - clrs_all * 0.35, (1 - clrs_all) * 0.35,
                             clrs_all*0.45, 1 - clrs_all * 0.45, (1 - clrs_all) * 0.45])
    #first color is dummy. track_id would never be 0, it starts from 1

    track_ids_gt=mots_helper.assign_ids(data, G, y_gt, train=True)
    track_ids_star = mots_helper.assign_ids(data, G, y_star, train=True)
    for i in rows:
        img_star = np.array(images[0][i].permute((1, 2, 0)))/255.
        img_gt = img_star.copy()


        clrs_gt = clrs_all[track_ids_gt[i]]
        clrs_star = clrs_all[track_ids_star[i]]

        for j, (mask, box) in enumerate(zip(data["masks"][i], data["boxes"][i])):

            if mask.dtype==torch.float:
                mask=mask>0.5
            img_star=mots_helper.apply_mask(img_star, mask, clrs_star[j], bb=box)
            img_gt = mots_helper.apply_mask(img_gt, mask, clrs_gt[j], bb=box)


        plt.subplot(n, 2, k)
        plt.imshow(img_gt)
        plt.axis('off')

        k += 1
        plt.subplot(n, 2, k)
        plt.imshow(img_star)
        plt.axis('off')

        k+=1
    #plt.suptitle("GT/Star")
    #plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return


#used in train
def plot_all(store, init_lmbda,  location="../toy_example2/",lr_lmbda=0.01, lr_theta=0.01 ):
    all_y_star, all_delta, all_L_star, all_L_GT, all_lmbda, all_iters, all_y_GT, all_grad_lmbda, mrcnn_loss,track_grad_theta_norm,mrcnn_grad_theta_norm= store["all_y_star"], store["all_delta"], store["all_L_star"], store["all_L_GT"], store["all_lmbda"], store["all_iters"], store["all_y_GT"], store["all_grad_lmbda"], store["mask_rcnn_loss"],store["track_grad_theta_norm"],store["mrcnn_grad_theta_norm"]
    plt.figure()

    plt.subplot(321)

    clrs = plt.cm.get_cmap('hsv', len(all_y_star[0])+1)#cm.rainbow(np.linspace(0, 1, len(all_y_star[0])))
    for i in range(0,len(all_y_star[0])):
        plt.plot(all_iters, [y_st[i] for y_st in all_y_star], label="y"+str(i)+"_star", color=clrs(i), alpha=0.7)
        plt.plot(all_iters, [y_GT[i] for y_GT in all_y_GT], linestyle='dashed', color=clrs(i), label="y"+str(i)+"_GT", alpha=0.7)
    #plt.plot(all_iters, [y2_GT for y_st in all_y_star], 'b--', label="y2_GT", alpha=0.7)

    plt.xlabel("Iterations")
    plt.ylabel("y")
    #plt.legend()

    plt.subplot(322)
    for i in range(0,len(all_lmbda[0])):
        plt.plot(all_iters, [th[i] for th in all_lmbda], label=str(i+1), alpha=0.7)
    plt.xlabel("Iterations")
    plt.ylabel("lmbda")
    #plt.legend()

    plt.subplot(323)
    #plt.plot(all_iters, list(np.array(all_delta) - np.array(all_L_star) + np.array(all_L_GT)), 'k', label="L",
             #alpha=0.7)
    #plt.plot(all_iters, [-i for i in all_L_star], 'r', label="F_star", alpha=0.7)
    #plt.plot(all_iters, [-i for i in all_L_GT], 'b', label="F_GT", alpha=0.7)
    plt.plot(all_iters, all_delta, 'g', label="delta", alpha=0.7)

    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.legend()

    plt.subplot(324)
    for i in range(0,len(all_grad_lmbda[0])):
        plt.plot(all_iters, [g[i] for g in all_grad_lmbda],label=str(i+1), alpha=0.7)

    plt.xlabel("Iterations")
    plt.ylabel("Grad_lmbda")
    #plt.legend()

    plt.subplot(325)
    div = int(len(all_iters) / len(store["epochs"]))
    plt.plot(store["epochs"], [torch.tensor(mrcnn_loss[(i - 1)*div:i * div]).sum()/div
                               for i in store["epochs"]],'r', label="seg_loss", alpha=0.7)
    plt.plot(store["epochs"], [torch.tensor(store["tracking_loss"][(i - 1)*div:i * div]).sum()/div
                               for i in store["epochs"]],'g', label="tracking_loss",alpha=0.7)
    #plt.plot(all_iters, [mrcnn_ls for mrcnn_ls in mrcnn_loss], 'r', label="mask_rcnn_loss", alpha=0.7)
    #plt.plot(all_iters, list(np.array(all_delta) - np.array(all_L_star) + np.array(all_L_GT)), 'g', label="tracking_loss",alpha=0.7)

    plt.xlabel("Epochs")
    plt.ylabel("Avg Losses")
    plt.legend()

    plt.subplot(326)
    div = int(len(all_iters) / len(store["epochs"]))
    plt.plot(store["epochs"], [torch.tensor(mrcnn_grad_theta_norm[(i - 1) * div:i * div]).sum() / div
                               for i in store["epochs"]], 'r', label="seg_loss", alpha=0.7)
    plt.plot(store["epochs"], [torch.tensor(track_grad_theta_norm[(i - 1)*div:i * div]).sum()/div
                               for i in store["epochs"]],'g', label="tracking_loss", alpha=0.7)


    plt.xlabel("Epochs")
    plt.ylabel("Norm of (grad_theta)")
    plt.legend()

    plt.tight_layout()
    plt.suptitle("init_lmbda: " + str(init_lmbda.tolist()), y=.995,
                 fontsize=8)

    plt.savefig(location+"/lr_lmbda" + str(lr_lmbda) + ".png", pad_inches=0.5)
    plt.close()

    plt.figure()
    total_mrcnn_loss=store["mask_rcnn_loss"]


    # plt.plot(store["epochs"], [torch.tensor(class_loss[(i - 1) * div:i * div]).sum() / div
    #                   for i in store["epochs"]], label="class_loss", alpha=0.7)
    # plt.plot(store["epochs"], [torch.tensor(box_reg[(i - 1) * div:i * div]).sum() / div
    #                   for i in store["epochs"]], label="box_reg",alpha=0.7)
    # plt.plot(store["epochs"], [torch.tensor(mask_loss[(i - 1) * div:i * div]).sum() / div
    #                   for i in store["epochs"]], label="mask_loss", alpha=0.7)

    div = int(len(all_iters) / len(store["epochs"]))
    plt.plot(store["epochs"], [torch.tensor(mrcnn_loss[(i - 1) * div:i * div]).sum() / div
                               for i in store["epochs"]], 'r', label="seg_loss", alpha=0.7)
    plt.plot(store["epochs"], [torch.tensor(store["tracking_loss"][(i - 1) * div:i * div]).sum() / div
                               for i in store["epochs"]], 'g', label="tracking_loss", alpha=0.7)

    #plt.plot(store["epochs"], [torch.tensor(total_mrcnn_loss[(i - 1) * div:i * div]).sum() / div
                      #for i in store["epochs"]], label="total_seg_loss",alpha=0.7)

    plt.xlabel("Epochs")
    plt.ylabel("Avg loss")
    plt.legend()

    plt.savefig(location+"/tracking_seg_loss.png")
    plt.close()


