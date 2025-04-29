from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, weight_decay=False, weight_decay_lambda=1e-8):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(
                    in_dim=size_list[i], 
                    out_dim=size_list[i + 1], 
                    weight_decay=weight_decay, 
                    weight_decay_lambda=weight_decay_lambda
                )
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        # print(f"Final norm of grad: {np.sqrt(np.sum(grads**2))}")
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.params['W'] = param_list[i + 2]['W']
            layer.params['b'] = param_list[i + 2]['b']
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplementedError
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append(
                    {
                        'W' : layer.params['W'], 
                        'b' : layer.params['b'], 
                        'weight_decay' : layer.weight_decay, 
                        'lambda' : layer.weight_decay_lambda
                    }
                )
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A convolutional neural network model for image classification.
    
    Default architecture for MNIST (28x28 grayscale images):
    - Conv1: 1->32 channels, 3x3 kernel
    - ReLU
    - Conv2: 32->64 channels, 3x3 kernel  
    - ReLU
    - Linear: 64*5*5 -> 10 (for MNIST classes)
    
    Parameters:
        conv_params (list[dict]): List of convolution layer parameters.
            Defaults to MNIST architecture if None.
            Each dict contains:
            - in_channels (int)
            - out_channels (int)
            - kernel_size (int)
            - stride (int, optional): Default 1
            - padding (int, optional): Default 0
            - lambda (float, optional): Weight decay lambda
            
        linear_size (tuple[int,int]): Input/output dims for linear layer.
            Default (64*5*5, 10) for MNIST.
            
        act_func (str): Activation function ('ReLU' supported)
    """
    def __init__(self, conv_params=None, num_classes=10, act_func='ReLU', Xavier=False):
        """Initialize CNN model with automatic dimension calculation.
        
        Args:
            conv_params: List of convolution layer parameters. Defaults to MNIST architecture.
            num_classes: Number of output classes. Default 10 for MNIST.
            act_func: Activation function ('ReLU' supported)
        """
        # Default MNIST architecture
        if conv_params is None:
            conv_params = [
                {'in_channels':1, 'out_channels':16, 'kernel_size':3},
                {'in_channels':16, 'out_channels':32, 'kernel_size':3}
            ]
            
        self.conv_params = conv_params
        self.num_classes = num_classes
        self.act_func = act_func
        self.layers = []
        
        # Track output dimensions through layers
        current_channels = 1  # Default input channels for MNIST
        current_height = 28   # MNIST input height
        current_width = 28    # MNIST input width
        
        # Build convolutional layers with optional pooling
        for params in conv_params:
            # Add conv layer
            layer = conv2D(
                in_channels=params['in_channels'],
                out_channels=params['out_channels'],
                kernel_size=params['kernel_size'],
                stride=params.get('stride', 1),
                padding=params.get('padding', 0),
                weight_decay='lambda' in params,
                weight_decay_lambda=params.get('lambda', 1e-8),
                Xavier=Xavier
            )
            self.layers.append(layer)
            current_channels = params['out_channels']
            
            # Update spatial dimensions
            current_height = (current_height - params['kernel_size'] + 2 * params.get('padding', 0)) // params.get('stride', 1) + 1
            current_width = (current_width - params['kernel_size'] + 2 * params.get('padding', 0)) // params.get('stride', 1) + 1
            
            # Add activation
            self.layers.append(ReLU())
            
            # Add pooling if specified for this layer
            if 'pool_params' in params:
                pool_size = params['pool_params'].get('pool_size', 2)
                stride = params['pool_params'].get('stride', pool_size)
                padding = params['pool_params'].get('padding', 0)
                self.layers.append(MaxPool2D(
                    pool_size=pool_size,
                    stride=stride,
                    padding=padding
                ))
                # Update spatial dimensions after pooling
                current_height = (current_height - pool_size + 2 * padding) // stride + 1
                current_width = (current_width - pool_size + 2 * padding) // stride + 1
        
        # Calculate flattened dimension for linear layer
        linear_input_dim = current_channels * current_height * current_width
        
        # Add Flatten layer before linear
        self.layers.append(Flatten())
        
        # Add final linear layer
        self.layers.append(Linear(
            in_dim=linear_input_dim,
            out_dim=num_classes,
            weight_decay='lambda' in conv_params[-1],
            weight_decay_lambda=conv_params[-1].get('lambda', 1e-8)
        ))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if not self.layers:
            raise ValueError('Model not initialized. Provide conv_params and linear_size')
            
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.conv_params = param_list[0]
        self.act_func = param_list[1]

        self.layers = []        # initialization of layers
        for i in range(len(self.conv_params)): 
            params = self.conv_params[i]  # the conv parameters
            layer = conv2D(
                in_channels=params['in_channels'],
                out_channels=params['out_channels'],
                kernel_size=params['kernel_size'], 
                weight_decay=params.get('weight_decay', False), 
                weight_decay_lambda=params.get('weight_decay_lambda', 1e-8)
            )
            # assignment of key parameters
            layer.W = param_list[i+2]['W']
            layer.b = param_list[i+2]['b']
            self.layers.append(layer)

            # ReLU
            if self.act_func == 'ReLU':
                self.layers.append(ReLU())
            else:
                raise NotImplementedError
            
            # maxpooling
            if 'pool_params' in params:
                pool_size = params['pool_params'].get('pool_size', 2)
                stride = params['pool_params'].get('stride', pool_size)
                padding = params['pool_params'].get('padding', 0)
                self.layers.append(MaxPool2D(
                    pool_size=pool_size,
                    stride=stride,
                    padding=padding
                ))
        
        # add flatten layer
        self.layers.append(Flatten())

        # add final linear layer
        linear_params = param_list[-1]
        in_dim, out_dim = linear_params['W'].shape
        layer = Linear(
            in_dim=in_dim,
            out_dim=out_dim,
            weight_decay=linear_params['weight_decay'],
            weight_decay_lambda=linear_params['weight_decay_lambda']
        )
        layer.params['W'] = linear_params['W']
        layer.params['b'] = linear_params['b']
        self.layers.append(layer)

        
    def save_model(self, save_path):
        param_list = [self.conv_params, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                if hasattr(layer, 'in_channels'):    # conv layer
                    param_list.append(
                        {
                            'W': layer.params['W'], 
                            'b': layer.params['b'], 
                            'weight_decay': layer.weight_decay, 
                            'weight_decay_lambda': layer.weight_decay_lambda, 
                            # 'in_channels': layer.in_channels, 
                            # 'out_channels': layer.out_channels, 
                            # 'kernel_size': layer.kernel_size, 
                            # 'stride': layer.stride, 
                            # 'padding': layer.padding, 
                            # 'pool_params': {'pool_size': 2, 'stride': 2}    # default config
                        }
                )
                else:           # linear layer
                    param_list.append(
                        {
                            'W': layer.params['W'], 
                            'b': layer.params['b'], 
                            'weight_decay': layer.weight_decay, 
                            'weight_decay_lambda': layer.weight_decay_lambda, 
                        }
                    )
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)