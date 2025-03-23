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

class SBLS:
    def __init__(self, input_size: int, output_size: int, simulation_steps: int, initital_feature_size: int, enhancement_nodes_per_window: int, initial_enhancement_window_num: int, output_is_one_hot: bool):
        self.input_size = input_size
        self.output_size = output_size
        self.simulation_steps = simulation_steps
        self.num_feature_nodes = initital_feature_size
        self.num_enhancement_windows = initial_enhancement_window_num
        self.enhancement_nodes_per_window = enhancement_nodes_per_window
        self.output_is_one_hot = output_is_one_hot
        self.training_samples = 0

        self.initialized = False    # Initialization state of the network. Upon creation, the final layer will still be random. Only when feeding data will W3 be solved to minimize the Least-Squares-Error. add_enhancement_nodes(), add_feature_nodes() and simpify() are only available when initialized = True.
        self.feature_layer_increased = False
        self.A_old = None           # stores all input values of trainin data for future optimization
        self.Z_post_old = None      # stores all Z_post values for adding new feature layer nodes (before aggregation)
        self.target_old = None           # stores all target values of training data for future optimization
        self.A_cross_old = None     # Most important member, the pseudo-inverse of matrix A. Is used to calculate the optimal W3 according to ridge regression based on training data. Upon adding new training data or adding nodes in the network, it can be incrementally updated

        # initialize weights and biases
        self.weight_range_W1 = math.sqrt(1/self.input_size)
        self.weight_range_W2 = math.sqrt(1/self.num_feature_nodes)
        self.weight_range_W3 = math.sqrt(1/(self.num_feature_nodes + self.num_enhancement_windows * self.enhancement_nodes_per_window))

        self.W1 = torch.zeros((self.input_size, self.num_feature_nodes)).uniform_(-self.weight_range_W1, self.weight_range_W1)
        self.B1 = torch.zeros((self.num_feature_nodes)).uniform_(-self.weight_range_W1, self.weight_range_W1)

        # The ehancement layer should be large (1000 nodes at very least). Because all datapoints should be passed at once (60.000 for MNIST e.g.), the enhancement layer values would consume 24GB RAM, which is too much for general hardware.
        # So instead, the enhancement layer is split into windows, which are calculated sequentially
        # self.W2 = torch.zeros((self.num_feature_nodes, self.num_enhancement_nodes)).uniform_(-self.weight_range_W2, self.weight_range_W2)
        self.W2_list = []
        self.B2_list = []

        for i in range(self.num_enhancement_windows):  
            self.W2_list.append(torch.zeros((self.num_feature_nodes, self.enhancement_nodes_per_window)).uniform_(-self.weight_range_W2, self.weight_range_W2))
            self.B2_list.append(torch.zeros(self.enhancement_nodes_per_window).uniform_(-self.weight_range_W2, self.weight_range_W2))

        self.W3 = torch.zeros((self.num_feature_nodes + self.num_enhancement_windows * self.enhancement_nodes_per_window, self.output_size)).uniform_(-self.weight_range_W3, self.weight_range_W3)
        # initialize spikegen and LIF

        self.spikegen = spikegen.rate

        self.lif = snn.Leaky(beta = 0.9, init_hidden=True)  # TODO: tune beta such that leaky and spikegen have ca. the same firing rate!


    def __call__(self, input: torch.Tensor) -> torch.Tensor:

        input = torch.reshape(input, (-1, self.input_size))

        _, A = self.__calc_Z_post_and_A(input)

        return A @ self.W3
        #return torch.softmax(A @ self.W3, 1)

    
    def __aggregate(self, tensor: torch.Tensor) -> torch.Tensor:
        tmp = tensor.sum(dim=0)

        min, _ = torch.min(tmp, 1, keepdim=True)
        max, _ = torch.max(tmp, 1, keepdim=True)

        return torch.nan_to_num((tmp - min) / (max - min))


    def __calc_Z_post_and_A(self, input: torch.Tensor) -> torch.Tensor:

        Z_post = self.spikegen(input @ self.W1 + self.B1, num_steps=self.simulation_steps)  # Z_pre = input @ self.W1 + self.B1 # no sparse autoencoding needed, spikegen is already sparse
        h_post_total_list = []

        for w2_window, b2_window in zip(self.W2_list, self.B2_list):

            # H_pre_window = torch.einsum('abc,cd->abd', Z_post, w2_window) + b2_window
            H_pre_window = Z_post @ w2_window + b2_window 

            h_post_window_list = []

            # feed timesteps sequentially into LIF-Neuron
            self.lif.reset_mem()
            for timestep in H_pre_window:
                h_post_window_list.append(self.lif(timestep))   # h_post = self.lif(elem)


            # stack timesteps along 0 dim to create new dimension (time x num_samples x layer_size), then aggregate to remove time dimension, then append to total list
            h_post_total_list.append(self.__aggregate(torch.stack(h_post_window_list, dim=0)))

        # concatenate Z_post and h_post_total_list along dim 1 (layer size) to create A
        A_list = [self.__aggregate(Z_post)]
        A_list.extend(h_post_total_list)

        return Z_post, torch.cat(A_list, 1)
    

    # check if a matrix is zero by employing a modified max norm which terminates prematurely with false if one elem absolute is > epsilon, true if all elems >= epsilon
    def __is_zero(self, matrix: torch.Tensor, epsilon: float) -> bool:
        for row in matrix:
            for elem in row:
                if abs(elem) > epsilon:
                    return False
                
        return True


    def add_new_data(self, data: list[torch.Tensor, torch.Tensor]):

        if self.feature_layer_increased:
            raise Exception("Cannot add new data after feature layer increase!")
        
        input, target = data

        input = torch.reshape(input, (-1, self.input_size))
        # compute A_x for new data
        Z_post, A_x = self.__calc_Z_post_and_A(input)
        if self.output_is_one_hot:
            # encode Y as a one-hot vector of targets
            Y = nn.functional.one_hot(target, num_classes=10).float()
        else:
            Y = target

        # if this is the first batch of data, A_new, A_cross_new and W3 are computed differently than if not
        if not self.initialized:
            # store Z_post, A and Y for future improvement and expansion of the network
            self.Z_post_old = Z_post
            self.A_old = A_x
            self.target_old = target

            # Calculate Moore-Penrose-Pseudoinverse, store it for future improvement, and use it to find least-squares-optimal W3 in the equation Y = A_x @ W_3
            self.A_cross_old = torch.pinverse(input=A_x, rcond=0.01)
            self.W3 = self.A_cross_old @ Y

            # set initialized to true to signal that the network is not ouputting random answers anymore. This makes add_enhancement_nodes(), add_feature_nodes() and simpify() available.
            self.initialized = True

        else:
            # append input matrix and target vector for future improvement and expansion of the network
            self.Z_post_old = torch.cat((self.Z_post_old, Z_post), dim=1)
            self.target_old = torch.cat((self.target_old, target), dim=0)
            self.A_old = torch.cat((self.A_old, A_x), dim=0)

            D_T = A_x @ self.A_cross_old

            # The below code calculates B = A_cross_old * D * (I + D_T * D)^-1
            B = torch.linalg.solve((torch.eye(D_T.shape[0]) + D_T @ D_T.T).T, (self.A_cross_old @ D_T.T).T).T
  
            self.A_cross_old = torch.cat((self.A_cross_old - B @ D_T, B), dim=1)

            self.W3 = self.W3 + B @ (Y - A_x @ self.W3)

        # increase training data count
        print(f"num samples before: {self.training_samples}")
        self.training_samples += input.shape[0]
        print(f"num samples after: {self.training_samples}")


    def add_feature_nodes(self, num_new_nodes: int):
        # check if network is initialized
        if not self.initialized:
            raise Exception("Cannot add nodes while network is not initialized! Initialize by adding some training data.")

        if self.feature_layer_increased:
            raise Exception("Can only expand feature nodes once!")

        

        self.feature_layer_increased = True
        

        
        
    def add_enhancement_windows(self, num_new_windows: int):
        # check if network is initialized
        if not self.initialized:
            raise Exception("Cannot add nodes while network is not initialized! Initialize by adding some training data.")

        if self.feature_layer_increased:
            raise Exception("Cannot add new enhancement node once feature layer was increased!")
        
        # Calculate H_post for new nodes
        H_post_new_list = []

        for i in range(num_new_windows):
            # generate new random weights between the feature layer and the new enhancement
            self.W2_list.append(torch.zeros((self.num_feature_nodes, self.enhancement_nodes_per_window)).uniform_(-self.weight_range_W2, self.weight_range_W2))
            self.B2_list.append(torch.zeros(self.enhancement_nodes_per_window).uniform_(-self.weight_range_W2, self.weight_range_W2))
            # calculate the output of the new nodes
            # h_pre_x = torch.einsum('abc,cd->abd', self.Z_post_old, self.W2_list[-1]) + self.B2_list[-1]
            h_pre_x = self.Z_post_old @ self.W2_list[-1] + self.B2_list[-1]

            h_post_x_list = []

            self.lif.reset_mem()
            for timestep in h_pre_x:    # timestep is (batch_size, enhancement_nodes_per_window)
                h_post_x_list.append(self.lif(timestep))

            h_post_x = torch.stack(h_post_x_list, dim=0)    # h_post_x is (timestep, batch_size, enhancement_nodes_per_window)

            agg = self.__aggregate(h_post_x)    # agg is (batch_size, enhancement_nodes_per_window)

            H_post_new_list.append(agg)  

        H_post_new = torch.cat(H_post_new_list, dim=1)  # H_post_new is (batch_size, num_new_windows * enhancement_nodes_per_window)

        # update the pseudoinverse incrementally
        D = self.A_cross_old @ H_post_new

        C = H_post_new - self.A_old @ D

        if self.__is_zero(C, 10e-10):
            print("is zero")
            B = torch.linalg.solve(torch.eye(D.shape[1]) + D @ D.T, D @ self.A_cross_old)

        else:
            print("is NOT zero")
            B = torch.pinverse(C)

        self.A_cross_old = torch.cat((self.A_cross_old - D @ B, B), dim=0)
        # self.A_cross_old = torch.pinverse(self.A_old)

        # update W3 incrementally
        if self.output_is_one_hot:
            Y = nn.functional.one_hot(self.target_old, num_classes=10).float()
        else:
            Y = self.target_old
        self.W3 = torch.cat((self.W3 - D @ B @ Y, B @ Y), dim=0)
        # self.W3 = self.A_cross_old @ Y

        # extend A_old with the new enhancement layer output
        self.A_old = torch.cat((self.A_old, H_post_new), dim=1) # TODO: this needs to go behind the calculation of C as it needs the old A_old

        # update enhancement window num
        self.num_enhancement_windows += num_new_windows