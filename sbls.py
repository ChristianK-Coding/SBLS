import math

import numpy as np
import matplotlib.pyplot as plot

import torch
import torch.nn as nn

import snntorch as snn
import snntorch.spikegen as spikegen


def show_img(img):
    img = np.array(img, dtype='float')
    # pixels = img.reshape((28, 28))
    plot.imshow(img, cmap='gray', aspect="auto", interpolation='none')
    plot.show()

class SBLS2:
    def __init__(self, input_size: int, output_size: int, simulation_steps: int, initital_feature_size: int, initial_enhancement_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.simulation_steps = simulation_steps
        self.feature_nodes = initital_feature_size
        self.enhancement_nodes = initial_enhancement_size
        self.training_samples = 0

        self.first_data_batch = True
        self.A_old  = None
        self.A_cross_old = None  # Most important member, the pseudo-inverse of matrix A. Is used to calculate the optimal W3 according to ridge regression based on training data. Upon adding new training data or adding nodes in the network, it can be incrementally updated

        # initialize weights and biases
        weight_range_W1 = math.sqrt(1/self.input_size)
        weight_range_W2 = math.sqrt(1/self.feature_nodes)
        weight_range_W3 = math.sqrt(1/(self.simulation_steps * (self.feature_nodes + self.enhancement_nodes)))

        self.W1 = torch.zeros((self.input_size, self.feature_nodes)).uniform_(-weight_range_W1, weight_range_W1)
        self.B1 = torch.zeros((self.feature_nodes)).uniform_(-weight_range_W1, weight_range_W1)

        self.W2 = torch.zeros((self.feature_nodes, self.enhancement_nodes)).uniform_(-weight_range_W2, weight_range_W2)
        self.B2 = torch.zeros(self.enhancement_nodes).uniform_(-weight_range_W2, weight_range_W2)

        self.W3 = torch.zeros((self.feature_nodes + self.enhancement_nodes, self.output_size)).uniform_(-weight_range_W3, weight_range_W3)

        # initialize spikegen and LIF

        self.spikegen = spikegen.rate

        self.lif = snn.Leaky(beta = 0.9, init_hidden=True)  # TODO: tune beta such that leaky and spikegen have ca. the same firing rate!


    def __call__(self, input: torch.Tensor) -> torch.Tensor:

        input = torch.reshape(input, (-1, self.input_size))

        A = self.__calc_A(input)

        # print("A")
        # print(A.shape)

        return torch.softmax(torch.mm(A, self.W3), 1)

    
    def __aggregate(self, A: torch.Tensor) -> torch.Tensor:
        tmp = torch.einsum("ijk -> jk", A)

        min, _ = torch.min(tmp, 1, keepdim=True)
        max, _ = torch.max(tmp, 1, keepdim=True)

        return (tmp - min) / (max - min)


    def __calc_A(self, input: torch.Tensor) -> torch.Tensor:

        # print("input")
        # print(input.shape)
        # show_img(input)

        Z_pre = torch.matmul(input, self.W1) + self.B1

        # print("Z_pre")
        # print(Z_pre.shape)
        # show_img(Z_pre)

        # normalize to range [0; 1]
        # min = Z_pre.min(1, keepdim = True)[0]
        # max = Z_pre.max(1, keepdim = True)[0]
        # Z_intermediate = (Z_pre - min) / (max - min)

        Z_post = self.spikegen(Z_pre, num_steps=self.simulation_steps)

        # print("Z_post")
        # print(Z_post.shape)

        # print(Z_post)

        H_pre = torch.einsum('abc,cd->abd', Z_post, self.W2) + self.B2

        # print("H_pre")
        # print(H_pre.shape)
        # show_img(H_pre.view(batch_size_train, -1))

        h_post_list = []

        for elem in H_pre:
            h_post = self.lif(elem)
            h_post_list.append(h_post)

        H_post = torch.stack(h_post_list, dim=0)

        # print("H_post")
        # print(H_post.shape)
         
        A = self.__aggregate(torch.cat((Z_post, H_post), 2))

        # show_img(A)

        return A



    def add_new_data(self, data: list[torch.Tensor, torch.Tensor]):
        
        input, target = data

        input = torch.reshape(input, (-1, self.input_size))

        # compute A_x for new data
        A_x = self.__calc_A(input)
        # encode Y as a one-hot vector of targets
        Y = nn.functional.one_hot(target, num_classes=10).float()

        # if this is the first batch of data, A_new, A_cross_new and W3 are computed differently than if not
        if self.first_data_batch:
            # Solve Y = A_x @ W3 for W3 with least squares (equivalent to W_3 = A_x^+(Pseudo-Inverse) @ Y)
            # self.A_old = A_x
            self.A_cross_old = torch.pinverse(A_x)
            self.W3 = self.A_cross_old @ Y

            self.first_data_batch = False

        else:
            # self.A_old = torch.cat((self.A_old, A_x), 0)

            D_T = torch.mm(A_x, self.A_cross_old)
            # The below code calculates B = A_cross_old * D * (I + D_T * D)^-1
            
            B = torch.linalg.solve((torch.eye(D_T.shape[0]) + D_T @ D_T.T).T, (self.A_cross_old @ D_T.T).T).T
  
            self.A_cross_old = torch.cat((self.A_cross_old - torch.mm(B, D_T), B), 1)  # TODO: check this again, it was late when I wrote this
            # print(f"A_cross_new: {A_cross_new.shape}")

            self.W3 = self.W3 + torch.mm(B, Y - torch.mm(A_x, self.W3))
            # print(f"W3: {self.W3.shape}")
            # print(self.W3[:,0])


        # print("W3")
        # print(self.W3.shape)
        # show_img(self.W3)
        # show_img(self.W3 == 0.0)

        # print(f"A_old: {self.A_old.size()}")
        # print(f"A_cross_old: {self.A_cross_old.size()}")

        # increase training data count
        print(f"num samples before: {self.training_samples}")
        self.training_samples += input.shape[0]
        print(f"num samples after: {self.training_samples}")
