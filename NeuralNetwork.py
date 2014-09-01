import numpy as np
import scipy.optimize
import copy

class NeuralNetwork(object):
    """Neural Network

    Create a simple feed-forward neural network with backpropagation and
    bias neurons in the hidden layers.

    Usage:
      NeuralNetwork(layout=list())
      NeuralNetwork.weights_init()     Nguyen Widrow initialisation of weights
      NeuralNetwork.weights_load(filename)   load weights from disk
      NeuralNetwork.fwd_propagation(input=list())   get NN response
    """
    def __init__(self, layout, has_bias=True):
        self.layout = copy.deepcopy(layout)
        self.has_bias = has_bias
        if has_bias:
            self.weights = [np.matrix(np.zeros((self.layout[i+1],(self.layout[i]+(i!=0))))) for \
                        i in range(len(self.layout)-1)]
        else:
            # no bias unit in input layer: (i!=0)
            self.weights = [np.matrix(np.zeros((self.layout[i+1],self.layout[i]))) for \
                        i in range(len(self.layout)-1)]


    def weights_init(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.__Nguyen_Widrow(self.weights[i])


    def __Nguyen_Widrow(self,w):
        shape = w.shape
        nw = np.matrix(np.random.random(shape) - 0.5)
        # number of hidden neurons
        h = sum(self.layout) - self.layout[0] - self.layout[-1]
        beta = 0.7 * np.power(h,(1.0/float(self.layout[0])))
        # euclidean norm
        n = np.linalg.norm(nw)
        nw = (beta * nw) / n
        return nw


    def weights_load(self, filepath):
        logger.info('load weights from file=%s' % filepath)
        self.weights = np.load(filepath)


    def weights_save(self, filepath):
        logger.info('save weights to file=%s' % filepath)
        np.save(filepath, self.weights)


    def __activation(self, z):
        return np.tanh(z)


    def __d_activation(self, a):
        return 1.0 - np.power(a, 2.0)


    def fwd_propagation(self, input):
        """
        propagate a single input forwards through the NN.
        This function is not vectorised and can only handle one set of input values at the time.
        Please use map(fwd_propagation, list of input observations)
        """
        if isinstance(input, np.matrix):
            raise ValueError("[NeuralNetwork][fwd_propagation] input must be list or array and not matrix!")

        if type(input) == list:
            if np.array(map(lambda x: type(x) != float, input)).any():
                raise ValueError("[NeuralNetwork][fwd_propagation] input must be list of floats!\n" + \
                                 "function is not vectorised! please use map(fwd_propagation, [[input],[input],[input]]) for multiple observations")
        elif "shape" in dir(input):
            if len(input.shape) > 1 and input.shape[0] > 1:
                raise ValueError("[NeuralNetwork][fwd_propagation] function is not vectorised! please use map for multiple input observations!")
        else:
            raise ValueError("[NeuralNetwork][fwd_propagation] input must be list or numpy array containing a single observation of inputs")

        if len(input) != self.layout[0]:
            raise ValueError("[NeuralNetwork][fwd_propagation] input dosn't comply with layout!\n" + \
                             "number of input values: %s, input nodes: %s\n" \
                             % (len(input), self.layout[0]))

        self.activation = []
        self.d_activation = []
        a1 = np.matrix(input).transpose()
        for i in range(len(self.layout)-1):
            a0 = a1
            if i>0 and self.has_bias:
                # no bias unit in input layer
                a0 = np.insert(a0,0,1,axis=0)
            self.activation.append(a0)
            z = self.weights[i] * a0
            if (i!=len(self.layout)-2):
                # the output neuron is linear as it is not a classifier but
                # function approximator
                a1 = self.__activation(z)
                self.d_activation.append(self.__d_activation(a1))
            else:
                # output layer is linear for function approximation
                a1 = z
                self.d_activation.append(np.matrix(np.ones(z.shape)))
        self.activation.append(a1)
        return self.activation[-1]


    def bck_propagation(self, target):
        if len(target) != self.layout[-1]:
            raise ValueError("[NeuralNetwork] the target values don't comply with the NN layout!\n" + \
                             "number of target values: %s, output nodes: %s\n" \
                              % (len(target), self.layout[-1]))

        delta = []
        nn_gradient = []
        nLayers = len(self.layout)
        d = self.activation[-1] - target
        for i in reversed(range(nLayers-1)):
            d = np.matrix((d * self.d_activation[i].transpose()).diagonal()).transpose()
            delta.insert(0, d)
            grad = d * self.activation[i].transpose()
            nn_gradient.insert(0, grad)
            if i==0: break
            d = self.weights[i].transpose() * delta[0]
            if self.has_bias:
                # bias neurons have no delta terms
                d = d[1:]
                
        return nn_gradient

        
    def nn_cost(self, nn_params, *args):
        """ The cost function for the optimizer.

        nn_params: flattened NN weights which need to be converted back
                   using self.__parse_flat_weights(nn_params)
        inputs: the neural network inputs for training
                type is numpy.array
                shape: columns are training sets and rows are features of each set
        targets: the neural network targets for training
                type is numpy.array
                shape: columns are training sets and rows are targets of each set
        gamma: regularisation
               type is float
        """
        inputs, targets, gamma = args
        self.__parse_flat_weights(nn_params)
        # the accumulated cost
        J = 0.0
        # loop over trainings set
        for i in range(len(inputs)):
            input = inputs[i,:]
            target = targets[i,:]
            activation = self.fwd_propagation(input)
            # cost function
            J += 1.0/float(2.0*len(inputs)) * np.sum( np.power((activation - target),2.0) )

        # regularised cost including cost of parameters
        for i in range(len(self.weights)):
            J += (gamma / float(2.0*len(inputs))) * np.sum(np.power(self.weights[i],2.0)) 
        
        return J

        
    def nn_gradient(self, nn_params, *args):
        """ The gradient for the optimizer.

        nn_params: flattened NN weights which need to be converted back
          using self.__parse_flat_weights(nn_params)
        inputs: the neural network inputs for training
          type is numpy.array
          shape: rows are training sets and columns are features of each set
        targets: the neural network targets for training
          type is numpy.array
          shape: rows are training sets and columns are targets of each set
        gamma: regularisation
          type is float
        """
        inputs, targets, gamma = args
        self.__parse_flat_weights(nn_params)
        # accumulated gradients
        if self.has_bias:
            gradients = [np.matrix(np.zeros((self.layout[i+1],(self.layout[i]+(i!=0))))) for \
                     i in range(len(self.layout)-1)]
        else:
            gradients = [np.matrix(np.zeros((self.layout[i+1],self.layout[i]))) for \
                     i in range(len(self.layout)-1)]

        # loop over trainings set
        for i in range(len(inputs)):
            input = inputs[i,:]
            target = targets[i,:]

            self.fwd_propagation(input)
            grad = self.bck_propagation(target)
            for j in range(len(grad)):
                gradients[j] += grad[j]

        for i in range(len(gradients)):
            if self.has_bias:
                gradients[i][int(i!=0):,:] += (gamma / float(len(inputs))) * self.weights[i][int(i!=0):,:]
            else:
                gradients[i][:,:] += (gamma / float(len(inputs))) * self.weights[i][:,:]

        
        return np.array(np.hstack([i.flatten() for i in gradients])).squeeze()


    def __parse_flat_weights(self, w):
        idx = 0
        for i in range(len(self.layout)-1):
            if self.has_bias:
                e = idx + ((self.layout[i]+(i!=0)) * self.layout[i+1])
                self.weights[i] = np.matrix(np.reshape(w[idx:e], (self.layout[i+1], \
                                                          (self.layout[i]+(i!=0)))))
            else:
                e = idx + (self.layout[i] * self.layout[i+1])
                self.weights[i] = np.matrix(np.reshape(w[idx:e], (self.layout[i+1], \
                                                          self.layout[i])))
                
            idx = e
            

    def train(self, inputs, targets, gamma=0.1, maxiter=20):
        """
        inputs: numpy.array
          shape: rows are observations and columns are features
        targets: numpy.array
          shape: rows are observations and columns are target
        gamma: optional float
          gamma is the regularisation parameter to prevent over-fitting
        maxiter: optional int
          the maximal number of iterations of the optimisation algorithm 
        """
        args = (inputs, targets, gamma)
        x0 = np.array(np.hstack([i.flatten() for i in self.weights])).squeeze()
        opt_weights, fopt, func_calls, grad_calls, warnflag, allvecs = \
                scipy.optimize.fmin_cg(self.nn_cost, x0, fprime=self.nn_gradient,
                                       args=args, maxiter=maxiter, disp=False, full_output=True, retall=True)
        
        self.__parse_flat_weights(opt_weights)

        
    def mae(self, inputs, targets):
        """
        calculate the MAE between targets and NN output
        inputs and targets are matrices or 2 dimensional arrays.
        each row is an observation. each column is a feature
        """
        error = 0.0
        for i in range(inputs.shape[0]):
            output = self.fwd_propagation(inputs[i,:])
            t = targets[i,:]
            error += abs(t-output)
        return error/float(inputs.shape[0])
