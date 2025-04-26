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
        output: [batch_size, out_dim]
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
        self.grads['W'] = self.input.T @ grad       # simplified formula
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)     # grad @ identity
        output = grad @ self.params['W'].T
        return output
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. This class implements a 2D convolutional layer
    with a specified number of input channels, output channels, kernel size,
    stride, padding, and weight initialization method.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        # Initialize the number of input and output channels, the kernel size, stride, and padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias using the provided method
        # Weights shape: [out_channels, in_channels, kernel_size, kernel_size]
        self.W = initialize_method(0, 1, (out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(0, 1, (out_channels, 1, 1))  # Bias shape: [out_channels, 1, 1]
        
        # Initialize gradients for weights and bias
        self.grads = {'W': None, 'b': None}
        
        # Weight decay parameters
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        # Store input for backpropagation
        self.X = None

    def __call__(self, X) -> np.ndarray:
        """
        This method allows the layer to be called as a function.
        It simply forwards the input to the forward method.
        
        Args:
            X (np.ndarray): Input data with shape [batch, channels, H, W]
        
        Returns:
            np.ndarray: Output data after convolution
        """
        return self.forward(X)

    def forward(self, X):
        """
        The forward pass of the 2D convolutional layer.
        
        Args:
            X (np.ndarray): Input data with shape [batch_size, in_channels, H, W]
        
        Returns:
            np.ndarray: Output data after convolution with shape [batch_size, out_channels, new_H, new_W]
        """
        # Store input for backpropagation
        self.X = X
        
        # Apply padding to the input
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        # Extract dimensions
        batch_size, _, H, W = X.shape
        new_H = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        new_W = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize the output matrix with zeros
        output = np.zeros((batch_size, self.out_channels, new_H, new_W))
        
        # Perform convolution
        for i in range(batch_size):
            for j in range(self.out_channels):
                for k in range(new_H):
                    for l in range(new_W):
                        h_start = k * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = l * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract the region from input image
                        X_slice = X_padded[i, :, h_start:h_end, w_start:w_end]
                        
                        # Perform element-wise multiplication and sum (convolution operation)
                        output[i, j, k, l] = np.sum(X_slice * self.W[j, :, :, :]) + self.b[j]
        
        return output

    def backward(self, grads):
        """
        The backward pass of the 2D convolutional layer.
        
        Args:
            grads (np.ndarray): Gradient of the loss with respect to the output of the layer, shape [batch_size, out_channels, new_H, new_W]
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the layer, shape [batch_size, in_channels, H, W]
        """
        # Extract dimensions
        batch_size, out_channels, new_H, new_W = grads.shape
        _, in_channels, H, W = self.X.shape
        kernel_size = self.kernel_size
        
        # Apply padding to the input
        X_padded = np.pad(self.X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        # Initialize gradients for weights, bias, and input
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX_padded = np.zeros_like(X_padded)
        
        # Perform backpropagation
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(new_H):
                    for l in range(new_W):
                        h_start = k * self.stride
                        h_end = h_start + kernel_size
                        w_start = l * self.stride
                        w_end = w_start + kernel_size
                        
                        # Extract the region from input image
                        X_slice = X_padded[i, :, h_start:h_end, w_start:w_end]
                        
                        # Add gradient of output to the gradient of weights and bias
                        dW[j, :, :, :] += X_slice * grads[i, j, k, l]
                        db[j] += grads[i, j, k, l]
                        
                        # Update the gradient of the input region
                        dX_padded[i, :, h_start:h_end, w_start:w_end] += self.W[j, :, :, :] * grads[i, j, k, l]
        
        # Extract the gradient of the input without padding
        dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        # Apply weight decay if enabled
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W
        
        # Store the gradients
        self.grads['W'] = dW
        self.grads['b'] = db
        
        return dX

    def clear_grad(self):
        """
        Clear the stored gradients.
        """
        self.grads = {'W': None, 'b': None}

        
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
            self.grads = (self.softmax_out - labels_vec) / batch_size     # simplified ver.
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