import numpy as np
import os
import pdb
import time
import math
import networkx as nx
import pycocotools.mask as cocomask
import torch
from multiprocessing import Pool
from external.HungarianMurty.HungarianMurty_lowmem import k_best_costs
from .utils import cost_functions as cf
from .utils import mots_helper

class AssignmentSpace():
    def __init__(self, args, detections, lmbda, train=True, parallelize=True):
        super().__init__()

        self.args=args
        self.lmbda=lmbda
        self.detections=detections
        self.train=train
        self.parallelize=parallelize
        self.method="Hungarian"


    def get_assignment_space(self, labels=[]): # get_phi_assignments
        phi_1_all=[]
        assignment_all=[]
        G = nx.DiGraph()
        y_GT=[]
        L_GT=0

        if self.args.mots:
            reid_shape=32
        else:
            reid_shape=32


        if self.parallelize:
            tc=time.time()

            cost_matrices, cost_matrices_iou, _ = cf.find_cost_matrix(self.args, {"boxes": self.detections.boxes[0:-1], "masks": self.detections.masks[0:-1], "reids": self.detections.reids[0:-1]},
                                           {"boxes": self.detections.boxes[1:], "masks": self.detections.masks[1:], "reids": self.detections.reids[1:]},
                                           self.lmbda, self.detections.images_shape[1:], optflow=self.detections.n_optflow_skip0, parallel=True, img_shape=[self.detections.images_shape[-2], self.detections.images_shape[-1]], reid_shape=reid_shape)

            print("cost_matrix_time", time.time() - tc)

            if self.args.second_order:
                cost_matrices_2nd_order,cost_matrices_iou_2nd_order,_ = cf.find_cost_matrix(self.args,
                    {"boxes": self.detections.boxes[0:-2], "masks": self.detections.masks[0:-2], "reids":self.detections.reids[0:-2]},
                    {"boxes": self.detections.boxes[2:], "masks": self.detections.masks[2:], "reids": self.detections.reids[2:]},
                    self.lmbda[int((self.lmbda.shape[0])/2):], self.detections.images_shape[1:], optflow=self.detections.n_optflow_skip1, parallel=True, img_shape=[self.detections.images_shape[-2], self.detections.images_shape[-1]], reid_shape=reid_shape)


            cost_matrix_list=[]
            cost_matrix_iou_list = []
            for i in range(len(cost_matrices)):
                #removing the nan's here
                cm=cost_matrices[i,:len(self.detections.boxes[i]),:len(self.detections.boxes[i+1])]
                cm_iou = cost_matrices_iou[i, :len(self.detections.boxes[i]), :len(self.detections.boxes[i + 1])]
                # if True in torch.isnan(cm):
                #     pdb.set_trace()
                cost_matrix_list.append(cm)
                cost_matrix_iou_list.append(cm_iou)

            if self.args.second_order:
                cost_matrix_2nd_order_list = []
                cost_matrix_iou_2nd_order_list = []
                for i in range(len(cost_matrices_2nd_order)):
                    cm2=cost_matrices_2nd_order[i, :len(self.detections.boxes[i]), :len(self.detections.boxes[i + 2])]
                    cm2_iou = cost_matrices_iou_2nd_order[i, :len(self.detections.boxes[i]), :len(self.detections.boxes[i + 2])]
                    # if True in torch.isnan(cm2):
                    #     pdb.set_trace()
                    cost_matrix_2nd_order_list.append(cm2)
                    cost_matrix_iou_2nd_order_list.append(cm2_iou)

        for i in range(0,len(self.detections.boxes)-1):
            time1 = time.time()

            if not self.parallelize:
                if not self.train and self.detections.n_optflow_skip0[0]!=[]:
                    flow0=-np.load(self.detections.n_optflow_skip0[i])
                    flow1 = -np.load(self.detections.n_optflow_skip1[i-1])
                else:
                    flow0 = self.detections.n_optflow_skip0[i]
                    flow1 = self.detections.n_optflow_skip1[i - 1]


                if self.args.mots:
                    masks_det_i=[dt.cuda(self.args.track_device) for dt in self.detections.masks[i]]
                    masks_det_i1=[dt.cuda(self.args.track_device) for dt in self.detections.masks[i+1]]

                else:
                    if len(self.detections.masks[i])>0:
                        masks_det_i = torch.tensor(cocomask.decode([dt for dt in self.detections.masks[i]])).permute(2,0,1).cuda(self.args.track_device)
                    else:
                        masks_det_i = torch.tensor([])
                    if len(self.detections.masks[i+1]) > 0:
                        masks_det_i1 = torch.tensor(cocomask.decode([dt for dt in self.detections.masks[i + 1]])).permute(2,0,1).cuda(self.args.track_device)
                    else:
                        masks_det_i1= torch.tensor([])

                cost_matrix, cost_matrix_iou,_= cf.find_cost_matrix(self.args, {"boxes":self.detections.boxes[i],"masks":masks_det_i,"reids":self.detections.reids[i]},
                                          {"boxes":self.detections.boxes[i+1],"masks":masks_det_i1,"reids":self.detections.reids[i+1]},
                                          self.lmbda, self.detections.images_shape[1:], optflow=flow0, reid_shape=reid_shape)
                if self.args.second_order and i > 0:
                    if self.args.mots:
                        masks_det_im1 = [dt.cuda(self.args.track_device) for dt in self.detections.masks[i-1]]
                        masks_det_i1 = [dt.cuda(self.args.track_device) for dt in self.detections.masks[i + 1]]

                    else:
                        if len(self.detections.masks[i-1])>0:
                            masks_det_im1 = torch.tensor(cocomask.decode([dt for dt in self.detections.masks[i-1]])).permute(2,0,1).cuda(self.args.track_device)
                        else:
                            masks_det_im1 = torch.tensor([])
                         #i minus 1
                        if len(self.detections.masks[i+1])>0:
                            masks_det_i1 = torch.tensor(cocomask.decode([dt for dt in self.detections.masks[i + 1]])).permute(2,0,1).cuda(self.args.track_device)
                        else:
                            masks_det_i1 = torch.tensor([])


                    cost_matrix_2nd_order, cost_matrix_iou_2nd_order,_ = cf.find_cost_matrix(self.args,
                        {"boxes": self.detections.boxes[i - 1], "masks": masks_det_im1,"reids":self.detections.reids[i-1]},
                        {"boxes": self.detections.boxes[i + 1], "masks": masks_det_i1,"reids":self.detections.reids[i+1]},
                        self.lmbda[int((self.lmbda.shape[0])/2):],self.detections.images_shape[1:], optflow=flow1, reid_shape=reid_shape)

            else:
                cost_matrix = cost_matrix_list[i]
                cost_matrix_iou = cost_matrix_iou_list[i]
                if self.args.second_order and i > 0:
                    cost_matrix_2nd_order = cost_matrix_2nd_order_list[i-1]
                    cost_matrix_iou_2nd_order = cost_matrix_iou_2nd_order_list[i - 1]
            #print("cm1: ", time.time() - time1)
            t2 = time.time()

                    #print("cm2: ", time.time() - t2)
            t3 = time.time()
            k = min(self.args.K,math.factorial(max(cost_matrix.shape[0], cost_matrix.shape[1])))
            assignment = []
            phi_1 =[]

            if self.method=="Hungarian" :#and cost_matrix.shape[0]>0 and cost_matrix.shape[1]>0
                #print(k,cost_matrix.shape[0],cost_matrix.shape[1])
                if True in torch.isnan(cost_matrix):
                    pdb.set_trace()
                #print(i, k, cost_matrix.shape)
                mx=max(cost_matrix.shape[0],cost_matrix.shape[1])
                cm_hung=np.zeros((mx,mx))+100000
                cm_hung[:cost_matrix.shape[0], :cost_matrix.shape[1]]=cost_matrix.detach().cpu().numpy()
                phi_1, assignment_hungarian = k_best_costs(k, cm_hung)
                for a_len in range(len(assignment_hungarian)):
                    a2=[]
                    a=assignment_hungarian[a_len]
                    for pair in a:
                        if pair[0] in list(range(cost_matrix.shape[0])) and pair[1] in list(range(cost_matrix.shape[1])):
                            a2.append(pair)
                    if a2!=[]:
                        assignment_hungarian[a_len]=np.stack(a2)
                    else:
                        assignment_hungarian[a_len] =np.empty((0,2), dtype=np.int64)
                print("hung time:", time.time()-t3)
                #phi_1=phi_1[:self.args.K]
                #assignment_hungarian=assignment_hungarian[:self.args.K]


                assignment=[]
                for a in assignment_hungarian:
                    zero_mat=torch.zeros((cost_matrix.shape)).cuda(self.args.track_device)
                    zero_mat[a[:,0],a[:,1]]=1
                    assignment.append(zero_mat)

                if self.train:
                    y_temp_gt, temp_assignment=get_y_temp_gt(labels[i], labels[i+1], assignment,  device=self.args.track_device)
                    if y_temp_gt==-1:
                        assignment.append(temp_assignment*1.)
                        y_temp_gt=len(assignment)-1
                        phi_1.append((cost_matrix*assignment[-1]).sum().item())

                    y_GT.append(y_temp_gt)

                    L_GT = L_GT +(cost_matrix*assignment[y_GT[i]]).sum() #phi_1[y_GT[i]]
                    # if cost_matrix.shape[0] != cost_matrix.shape[1]:
                    #     auxiliary_cost = (max(cost_matrix.shape) - min(cost_matrix.shape)) * max_cost * (
                    #         (self.lmbda[:int((self.lmbda.shape[0]) / 2)]).sum())
                    #
                    #     L_GT = L_GT + auxiliary_cost

                    if i>0 and self.args.second_order and len(assignment_all[i-1][y_GT[i-1]]) and len(assignment[y_GT[i]]):

                        assignment_matrix_2nd_order = assignment_all[i - 1][y_GT[i - 1]] @ assignment[y_GT[i]]

                        a1 = assignment_all[i - 1][y_GT[i - 1]]
                        a2 = assignment[y_GT[i]]
                        auxiliary_cost = 0
                        if (a1.shape[0] > a1.shape[1] and a2.shape[0] < a2.shape[1]):
                            #counts = max(a1.shape[0], a2.shape[1]) - a2.shape[0]
                            assignment_matrix_2nd_order, auxiliary_cost = mots_helper.hung_2nd_order(assignment_matrix_2nd_order,
                                                                                         cost_matrix_2nd_order.detach())
                            #auxiliary_cost = counts * max_cost * ((self.lmbda[int((self.lmbda.shape[0]) / 2):]).sum())

                        L_GT = L_GT + (cost_matrix_2nd_order * assignment_matrix_2nd_order).sum() + auxiliary_cost

                else:
                    y_GT.append(-1)
                    L_GT=torch.tensor([0.])

            else:# cost_matrix.shape[0]>0 and cost_matrix.shape[1]>0:
                phi_1, assignment = k_best_costs_nlp(k, cost_matrix.copy())
            #print("hung: ", time.time() - t3)
            #print("first half: ", time.time() - time1)
            time2 = time.time()
            for i_phi, _ in enumerate(phi_1):
                val_phi=(assignment[i_phi]*cost_matrix).sum()
                # if cost_matrix.shape[0]!=cost_matrix.shape[1]:
                #     auxiliary_cost = (max(cost_matrix.shape) - min(cost_matrix.shape)) * max_cost * (
                #         (self.lmbda[:int((self.lmbda.shape[0]) / 2)]).sum())
                #     val_phi=val_phi+auxiliary_cost
                #     phi_1[i_phi]=phi_1[i_phi]+auxiliary_cost.item()

                #a_p = np.array(np.where(np.round(assignment[i_phi]))).transpose()
                G.add_node(str(i) + "_" + str(i_phi))
                #print(val_phi)
                G.nodes[str(i) + "_" + str(i_phi)]["node_cost"] = val_phi.type(torch.float64)
                G.nodes[str(i) + "_" + str(i_phi)]["cost_matrix"] = cost_matrix
                G.nodes[str(i) + "_" + str(i_phi)]["cost_matrix_iou"] = np.array(cost_matrix_iou.detach().cpu())
                # cost between detections i, i+1
                if self.args.second_order and i > 0:
                    G.nodes[str(i) + "_" + str(i_phi)]["cost_matrix_iou_2nd_order"] = np.array(cost_matrix_iou_2nd_order.detach().cpu())
                    # cost between detections i-1, i+1
                if self.train:
                    G.nodes[str(i) + "_" + str(i_phi)]["delta"] = abs(assignment[y_temp_gt]-assignment[i_phi]).sum()
                #cost_grad_lmbda=np.sum(np.array([cost_grad_lmbda_matrix[i2[0], i2[1]] for i2 in assignment_positions]),axis=0)  # np.stack([(cost_grad_lmbda_matrix[:,:,0]*assignment[i_phi]).sum(),(cost_grad_lmbda_matrix[:,:,1]*assignment[i_phi]).sum()])
                #G.nodes[str(i) + "_" + str(i_phi)]["cost_grad_lmbda"] = cost_grad_lmbda
                #temp_theta=(asgn_phi.reshape(2, 2, 1) * cost_grad_theta_matrix.reshape(2, 2, 8)).reshape(2,2,2,2,2)#+assignment[i_phi].reshape(2, 2, 1)*cost_matrix.reshape(2,2,1)
                G.nodes[str(i) + "_" + str(i_phi)]["node_assignment"] = assignment[i_phi].clone()
            phi_1_all.append(phi_1)
            assignment_all.append(assignment)


            if i>0:
                i2=i-1
                a=assignment_all[i2]


                for a_iter in range(0,len(a)):
                    t_2nd = time.time()
                    for a_next_iter in range(0, len(assignment_all[i2+1])):
                        phi_2 = torch.tensor(0.).cuda(self.args.track_device)
                        phi_1 = (G.nodes[str(i2) + "_" + str(a_iter)]["node_cost"] / 2 +
                                 G.nodes[str(i2 + 1) + "_" + str(a_next_iter)][
                                     "node_cost"]) / 2

                        if self.args.second_order and len(G.nodes[str(i2 + 1) + "_" + str(a_next_iter)][
                                          "node_assignment"]) and len(G.nodes[str(i2) + "_" + str(a_iter)]["node_assignment"]):

                            assignment_matrix_2nd_order = (G.nodes[str(i2) + "_" + str(a_iter)]["node_assignment"] @
                                                           G.nodes[str(i2 + 1) + "_" + str(a_next_iter)][
                                                               "node_assignment"])
                            a1 = G.nodes[str(i2) + "_" + str(a_iter)]["node_assignment"]
                            a2 = G.nodes[str(i2 + 1) + "_" + str(a_next_iter)]["node_assignment"]
                            auxiliary_cost = 0
                            if (a1.shape[0] > a1.shape[1] and a2.shape[0] < a2.shape[1]):
                                counts = max(a1.shape[0], a2.shape[1]) - a2.shape[0]

                                assignment_matrix_2nd_order, auxiliary_cost=mots_helper.hung_2nd_order(assignment_matrix_2nd_order, cost_matrix_2nd_order)
                                #auxiliary_cost = counts * max_cost * ((lmbda[int((lmbda.shape[0]) / 2):]).sum())
                            # 3 is the maximum cost possible for a pair of detections, iou_cost=1, app_cost=1, dist_cost=1

                            phi_2 = (assignment_matrix_2nd_order * cost_matrix_2nd_order).sum()+auxiliary_cost

                        L = phi_1 + phi_2

                        if self.train:

                            delta = (G.nodes[str(i2) + "_" + str(a_iter)]["delta"] / 2 +
                                 G.nodes[str(i2 + 1) + "_" + str(a_next_iter)][
                                     "delta"]) / 2
                        else:
                            delta=0

                        G.add_edge(str(i2) + "_" + str(a_iter), str(i2 + 1) + "_" + str(a_next_iter), weight= - delta + L)
                        attr = {
                            (str(i2) + "_" + str(a_iter), str(i2 + 1) + "_" + str(a_next_iter)): {'second_order_cost': phi_2}}
                        nx.set_edge_attributes(G, attr)
                        del L, phi_2, phi_1



        if len(list(G.edges))>0:
            start = list(G.edges)[0][0][:list(G.edges)[0][0].find("_")]
            stop = list(G.edges)[-1][1][:list(G.edges)[-1][1].find("_")] + "_0"
            stop = stop[:stop.find("_")]
        else:
            start = '0'
            stop = '0'
        G.add_node("start")
        G.add_node("end")

        for n in list(G.nodes):
            if n[:n.find("_")] == start:
                L = G.nodes[n]["node_cost"] / 2
                if self.train:
                    delta = G.nodes[n]["delta"] / 2
                else:
                    delta=0
                G.add_edge("start", n, weight=-delta + L)
            if n[:n.find("_")] == stop:
                L = G.nodes[n]["node_cost"] / 2
                if self.train:
                    delta = G.nodes[n]["delta"] / 2
                else:
                    delta=0
                G.add_edge(n, "end",weight=-delta + L)

        min_val = np.array([data.item() for e1, e2, data in G.edges(data="weight")]).min()
        #print("G min val:", min_val)
        if min_val < 0:
            for e in list(G.edges):
                if e[0]!="start" or e[1]!="end":
                    G.edges[e]["weight"] = G.edges[e]["weight"] - min_val
                else:
                    G.edges[e]["weight"] = G.edges[e]["weight"] - min_val/2

            min_val = np.array([data.item() for e1, e2, data in G.edges(data="weight")]).min()
            #print("G min val (modified):", min_val)

        self.G = G
        self.L_GT = L_GT
        self.y_GT = y_GT

        return

    def get_best_path(self): # get_star_GT

        optimized_path=nx.dijkstra_path(self.G, "start", "end")#["0_"+str(best[0][0])]#, "1_"+str(best[1][0])]
        optimized_path.remove("start")
        optimized_path.remove("end")

        self.y_star=[int(i[i.find("_") + 1:]) for i in optimized_path]

        phi_1_star = torch.stack([self.G.nodes[i]["node_cost"].type(torch.float64) for i in optimized_path]).sum()
        phi_2_star = 0.
        if self.args.second_order:
            phi_2_star = torch.stack([self.G.edges[i, j]["second_order_cost"].type(torch.float64) for _, (i, j) in enumerate(zip(optimized_path[:-1], optimized_path[1:]))]).sum()

        self.L_star = (phi_1_star + phi_2_star).cuda(self.args.track_device)

        if self.train:
            self.delta_star=np.array([self.G.nodes[i]["delta"] for i in optimized_path]).sum()
        else:
            self.delta_star=0

        return

    def get_permutations_cost(self):
        self.interval=self.detections.t2-self.detections.t1-1
        lists=[]
        cost_lists=[]

        for i in range(self.interval):
            lists.append([k for k in list(self.G.nodes())  if k.startswith(str(i)+"_")])
            cost_lists.append([self.G.nodes[k]["node_cost"] for k in list(self.G.nodes())  if k.startswith(str(i)+"_")])


        self.alpha_names=[(a, b, c, d) for a in lists[0] for b in lists[1] for c in lists[2] for d in lists[3]]
        t1=time.time()
        self.alpha_costs=torch.stack([torch.tensor([a, b, c, d]) for a in cost_lists[0] for b in cost_lists[1] for c in cost_lists[2] for d in cost_lists[3]])
        print("alpha_costs:",time.time()-t1)
        t1=time.time()
        self.alphas=AlphaSpace(self, np.stack(self.alpha_names), self.alpha_costs)

        #self.alphas.phi = list(map(get_phi, [self.alphas for k in range(4)], [1,2,3,4]))
        #self.alphas.phi = [get_phi(self.alphas, 1)]#list(map(get_phi, [self.alphas for k in range(1)], [1]))
        self.alphas.get_phi([1,2,3,4])



        return

    def get_best_permutation(self):

        self.total_cost=self.alphas.phi

        best_permutation=self.alpha_names[torch.argmin(self.total_cost)]
        self.y_star=[int(b[b.find("_")+1:]) for b in best_permutation]
        return


class AlphaSpace():
    def __init__(self, assignments, alpha_names, alpha_costs):
        super().__init__()

        t1=time.time()
        self.node_cost=alpha_costs

        self.node_assignment=[torch.stack([assignments.G.nodes[k[i]]["node_assignment"] for k in alpha_names]) for i in range(4)]

        self.assignments=assignments
        self.alpha_names=alpha_names

    def get_phi(self, orders=None): #currently supports 1,2,3,4


        phi0=torch.sum(self.node_cost, dim=1).cuda()

        det={"boxes": self.assignments.detections.boxes[0:-2]+ self.assignments.detections.boxes[0:-3]+ self.assignments.detections.boxes[0:-4],
                    "masks": self.assignments.detections.masks[0:-2]+ self.assignments.detections.masks[0:-3]+ self.assignments.detections.masks[0:-4],
                    "reids":self.assignments.detections.reids[0:-2]+ self.assignments.detections.reids[0:-3]+ self.assignments.detections.reids[0:-4]}
        det_next={"boxes": self.assignments.detections.boxes[2:]+ self.assignments.detections.boxes[3:]+self.assignments.detections.boxes[4:],
                    "masks": self.assignments.detections.masks[2:]+ self.assignments.detections.masks[3:]+self.assignments.detections.masks[4:],
                    "reids": self.assignments.detections.reids[2:]+ self.assignments.detections.reids[3:]+self.assignments.detections.reids[4:]}
        shapes=[(len(a), len(b)) for _,(a,b) in enumerate(zip(det["boxes"], det_next["boxes"]))]


        t1=time.time()
        cost_matrices, _, _ = cf.find_cost_matrix(self.assignments.args, det, det_next, self.assignments.lmbda[int((self.assignments.lmbda.shape[0])/2):], self.assignments.detections.images_shape[1:], optflow=self.assignments.detections.n_optflow_skip1, parallel=True, img_shape=[self.assignments.detections.images_shape[-2], self.assignments.detections.images_shape[-1]], reid_shape=32)
        print("h order cm: ", time.time()-t1)
        t1=time.time()
        #cost_matrices=[cost_matrices_temp[i, :len(self.assignments.detections.boxes[i]), :len(self.assignments.detections.boxes[i + order])] for i in range(len(cost_matrices_temp))]
        assignment_matrices=get_assignment_matrices(self.node_assignment)


        phi = torch.stack([(a*c[:s[0],:s[1]]).sum(-1).sum(-1) for _, (a,c,s) in enumerate(zip(assignment_matrices, cost_matrices, shapes))]).sum(0)
        aux = torch.stack([(1-a.sum(-1)).sum(-1)*100 for a in assignment_matrices]).sum(0)

        t1=time.time()
        #phi=torch.tensor([sum([(a*c).sum().sum() for _, (a,c) in enumerate(zip(am, cost_matrices))])+ sum([(1-a.sum(0)).sum()*100 for a in am])  for am in assignment_matrices])
        # AUX IS ADDED
        self.phi=phi0+phi+aux

        return



def get_assignment_matrices(alpha_node_assignment, lngth=3):
    assign_mat=[alpha_node_assignment[i]@alpha_node_assignment[i+1] for i in range(lngth)]+[alpha_node_assignment[i]@alpha_node_assignment[i+1]@alpha_node_assignment[i+2] for i in range(lngth-1)]+[alpha_node_assignment[i]@alpha_node_assignment[i+1]@alpha_node_assignment[i+2]@alpha_node_assignment[i+3] for i in range(lngth-2)]

    return assign_mat
