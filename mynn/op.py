from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, moment = False, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # self.W = initialize_method(size=(in_dim, out_dim))
        # self.W /= np.sqrt(np.sum(self.W**2))                # normalization 
        # self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        if moment:
            self.velocity = {'W' : np.zeros((in_dim, out_dim)), 'b' : np.zeros((1, out_dim))}
        self.input = None # Record the input for backward process.

        self.params = {}
        self.params['W'] = initialize_method(size=(in_dim, out_dim))
        self.params['b'] = initialize_method(size=(1, out_dim))

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X                          # record the input for using in backward method
        output = X @ self.params['W'] + self.params['b']            # affine transformation
        return output      # implicit broadcasting of b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # method 1 to compute W: memory-friendly
        # tmp = np.zeros_like(self.W)
        # batch_size, out_dim = grad.shape
        # for i in range(batch_size):
        #     for j in range(out_dim):
        #         tmp[:, j] += grad[i, j] * self.input[i, ]        # X = self.input
        self.grads['W'] = self.input.T @ grad       # simplified formula
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)     # grad @ identity
        output = grad @ self.params['W'].T
        return output
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        pass

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.has_softmax = True     # enable by default
        self.max_classes = max_classes
        self.predicts = None        # the input
        self.labels = None          # the labels
        self.grads = None           # the grads of the layer
        self.softmax_out = None     # the output of softmax layer


    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        batch_size, = labels.shape
        if not self.has_softmax:    # do not need softmax
            output = np.mean(-np.log(predicts[np.arange(batch_size), labels]))
        else:                       # pass an extra function
            self.softmax_out = softmax(predicts)    # [batch_size, D]
            output = np.mean(-np.log(self.softmax_out[np.arange(batch_size), labels] + 1e-10))
        return output
    
    def backward(self):
        # first compute the grads from the loss to the input
        # create a set of one-hot vectors first
        batch_size = len(self.labels)
        labels_vec = np.zeros((batch_size, self.max_classes))
        labels_vec[np.arange(batch_size), self.labels] = 1   
        if not self.has_softmax:
            # no softmax layer, simple gradient
            self.grads = labels_vec / (-self.predicts) / batch_size
        else:
            # two steps
            # self.grads = np.empty_like(self.predicts)
            self.grads = (self.softmax_out - labels_vec) / batch_size     # simplified ver.
            # for i in range(batch_size):     # use the ith sample
            #     # softmax_grad = np.empty((self.max_classes, self.max_classes))
            #     # x_max = np.max(self.predicts[i, :]) 
            #     # x_exp = np.exp(self.predicts[i, :] - x_max)     # one-dim vector
            #     # partition = np.sum(x_exp)       # scalar
            #     output = self.softmax_out[i, :]
            #     softmax_grad = np.diag(output) - np.outer(output, output)
            #     self.grads[i, :] = self.grads[i, :] @ softmax_grad
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition