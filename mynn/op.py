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
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.grads = {'W' : None, 'b' : None}
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
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']
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
        super().__init__()
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

        # used in updating parameters in optimizer.py
        self.params = {'W': self.W, 'b': self.b}
        
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
        
        # Initialize the output matrix
        output = np.zeros((batch_size, self.out_channels, new_H, new_W))
        
        # Vectorized convolution implementation
        for k in range(new_H):
            for l in range(new_W):
                h_start = k * self.stride
                h_end = h_start + self.kernel_size
                w_start = l * self.stride
                w_end = w_start + self.kernel_size
                
                # Extract all batch and channel slices at once
                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]
                
                # Vectorized computation across output channels
                # output[:, :, k, l] = np.tensordot(X_slice, self.W, axes=([1,2,3],[1,2,3])) + self.b.squeeze()
                output[:, :, k, l] = np.einsum('ijkl, mjkl -> im', X_slice, self.W) + self.b.squeeze()
        
        return output

    def backward(self, grads):
        """
        The backward pass of the 2D convolutional layer.
        
        Args:
            grads (np.ndarray): Gradient of the loss with respect to the output of the layer, 
            shape [batch_size, out_channels, new_H, new_W]
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the layer, 
            shape [batch_size, in_channels, H, W]
        """
        # Extract dimensions
        batch_size, out_channels, new_H, new_W = grads.shape
        kernel_size = self.kernel_size
        
        # Apply padding to the input
        X_padded = np.pad(self.X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        # Initialize gradients
        dW = np.zeros_like(self.W)
        db = np.sum(grads, axis=(0,2,3)).reshape(-1, 1, 1)  # Sum over batch and spatial dims
        dX_padded = np.zeros_like(X_padded)
        
        # Vectorized backpropagation
        for k in range(new_H):
            for l in range(new_W):
                h_start = k * self.stride
                h_end = h_start + kernel_size
                w_start = l * self.stride
                w_end = w_start + kernel_size
                
                # Get all input regions at once
                X_regions = X_padded[:, :, h_start:h_end, w_start:w_end]
                
                # Vectorized weight gradient update
                dW += np.tensordot(grads[:, :, k, l], X_regions, axes=([0],[0]))
                
                # Vectorized input gradient update
                dX_padded[:, :, h_start:h_end, w_start:w_end] += np.tensordot(
                    grads[:, :, k, l], self.W, axes=([1],[0])
                )
        
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
    
class MaxPool2D(Layer):
    """
    2D Max Pooling layer that performs downsampling by taking the maximum value
    over a spatial window for each channel.
    
    Args:
        pool_size (int): Size of the pooling window (square)
        stride (int, optional): Stride of the pooling operation. Defaults to pool_size.
        padding (int, optional): Zero padding to add around input. Default 0.
    """
    def __init__(self, pool_size=2, stride=None, padding=0):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding
        self.input = None
        self.max_indices = None  # To store locations of max values for backprop
        self.optimizable = False  # No learnable parameters

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        Forward pass of max pooling.
        
        Args:
            X (np.ndarray): Input tensor of shape [batch, channels, height, width]
            
        Returns:
            np.ndarray: Output tensor after max pooling
        """
        self.input = X
        
        # Apply padding if needed
        if self.padding > 0:
            X = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), 
                          (self.padding, self.padding)), mode='constant')
        
        batch, channels, height, width = X.shape
        out_h = (height - self.pool_size) // self.stride + 1
        out_w = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch, channels, out_h, out_w))
        self.max_indices = np.zeros((batch, channels, out_h, out_w, 2), dtype=np.int32)
        
        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride 
                        w_end = w_start + self.pool_size
                        
                        window = X[b, c, h_start:h_end, w_start:w_end]
                        # output[b, c, i, j] = np.max(window)
                        
                        # Store location of max value for backprop
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indices[b, c, i, j] = [h_start + max_idx[0], 
                                                       w_start + max_idx[1]]
                        output[b, c, i, j] = window[max_idx]
        
        return output

    def backward(self, grads):
        """
        Backward pass of max pooling.
        
        Args:
            grads (np.ndarray): Gradient of loss w.r.t. output, shape [batch, channels, out_h, out_w]
            
        Returns:
            np.ndarray: Gradient of loss w.r.t. input
        """
        batch, channels, out_h, out_w = grads.shape
        _, _, height, width = self.input.shape
        
        # Initialize output gradient with zeros
        dX = np.zeros_like(self.input)
        
        # Apply padding if needed
        if self.padding > 0:
            pad_width = ((0,0), (0,0), (self.padding, self.padding), 
                        (self.padding, self.padding))
            dX = np.pad(dX, pad_width, mode='constant')
        
        # Distribute gradients only to max locations
        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h, w = self.max_indices[b, c, i, j]
                        dX[b, c, h, w] += grads[b, c, i, j]
        
        # Remove padding if applied
        if self.padding > 0:
            dX = dX[:, :, self.padding:-self.padding, self.padding:-self.padding]
            
        return dX

class Flatten(Layer):
    """
    Flatten layer that reshapes input to 2D (batch_size, -1)
    
    Forward pass:
        Y = reshape(X, (batch_size, -1))
        where X is input tensor of shape (batch_size, ...)
    
    Backward pass:
        dL/dX = reshape(dL/dY, original_input_shape)
        Since flattening is just a reshape operation,
        the gradient flows backward by reshaping to original dimensions
    """
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None
        
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
        
    def backward(self, grad):
        # Gradient flows backward by reshaping to original input dimensions
        # This is mathematically correct since:
        # ∂L/∂X_ijk... = ∂L/∂Y_ij where Y = flatten(X)
        return grad.reshape(self.input_shape)

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