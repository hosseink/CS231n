import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class NLayerConvNet(object):
  """
  A multiple-layer convolutional network with the following architecture:
  
  [conv - relu - bn - 2x2 max pool] * N - [affine - bn - relu] * M - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), N = 1, M = 1, num_filters=32, 
               filter_sizes=7, hidden_dims=100, num_classes=10, 
               use_batchnorm = False, weight_scale=1e-3, reg=0.0, dtype=np.float32):

    """
    Initialize a new network.
    
    Inputs:
    - N: number of CNN layers
    - M: number of fully connected affine layers
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: A list of number of filters to use in the convolutional layer
    - filter_sizes: A list of size of filters to use in the convolutional layer
    - hidden_dims: A list of number of units to use in the fully-connected hidden layers
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.N = N
    self.M = M
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    if not type(num_filters)==list:
      num_filters = [num_filters for _ in range(N)]
    if not type(filter_sizes)==list:
      filter_sizes = [filter_sizes for _ in range(N)]
    if not type(hidden_dims)==list:
      hidden_dims = [hidden_dims for _ in range(M)]
    
    self.filter_size = filter_sizes
    self.num_filters = num_filters
    self.hidden_dims = hidden_dims

    C, H, W = input_dim
    cnn_sizes = [(num_filters[0], C, filter_sizes[0], filter_sizes[0])]
    cnn_sizes.extend([(num_filters[i], num_filters[i-1], filter_sizes[i], 
                                                filter_sizes[i]) for i in range(1,N)])
    
    for i in range(N):
      self.params['CW' + str(i+1)] = weight_scale * \
                    np.random.randn(cnn_sizes[i][0], cnn_sizes[i][1], 
                                     cnn_sizes[i][2], cnn_sizes[i][2])
      self.params['Cb' + str(i+1)] = np.zeros(num_filters[i])
      if self.use_batchnorm:
        self.params['Cgamma' + str(i+1)] = np.ones(num_filters[i])
        self.params['Cbeta' + str(i+1)] = np.zeros(num_filters[i])

   
    HO = H/2**N
    WO = W/2**N
    affine_sizes = [(HO*WO*num_filters[-1], hidden_dims[0])]
    affine_sizes.extend([(hidden_dims[i], hidden_dims[i+1]) for i in range(M-1)])

    for i in range(M):
      self.params['AW' + str(i+1)] = weight_scale * \
							np.random.randn(affine_sizes[i][0], affine_sizes[i][1] )
      self.params['Ab' + str(i+1)] = np.zeros(affine_sizes[i][1])
      if self.use_batchnorm:
        self.params['Agamma' + str(i+1)] = np.ones(affine_sizes[i][1])
        self.params['Abeta' + str(i+1)] = np.zeros(affine_sizes[i][1])

    self.params['W'] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
    self.params['b'] = np.zeros(num_classes)

    
    if self.use_batchnorm:
      self.Cbn_param = [{'mode': 'train'} for _ in range(N)]
      self.Abn_param = [{'mode': 'train'} for _ in range(M)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    N = self.N
    M = self.M
    filter_sizes = self.filter_size 
    num_filters = self.num_filters
    hidden_dims = self.hidden_dims

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    Ccache_list = []
    Acache_list = []
    out = X 
    for i in range(N):
      W, b = self.params['CW' + str(i+1)], self.params['Cb'+str(i+1)]
      conv_param = {'stride': 1, 'pad': (filter_sizes[i] - 1) / 2}
      pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
      if self.use_batchnorm:
        gamma, beta = self.params['Cgamma' + str(i+1)], self.params['Cbeta'+str(i+1)]
        out, cache = conv_batchnorm_relu_pool_forward(out, 
						W, b, gamma, beta, conv_param, self.Cbn_param[i], pool_param)
      else:  
        out, cache = conv_relu_pool_forward(X, W, b, conv_param, pool_param)
      Ccache_list.append(cache)

    for i in range(M):
      W, b = self.params['AW' + str(i+1)], self.params['Ab'+str(i+1)]
      if self.use_batchnorm:
        gamma, beta = self.params['Agamma' + str(i+1)], self.params['Abeta'+str(i+1)]
        out, cache = affine_batchnorm_relu_forward(out, W, b, 
												gamma, beta, self.Abn_param[i])
      else:  
        out, cache = affine_relu_forward(out, W, b)
      Acache_list.append(cache)

    W, b = self.params['W'], self.params['b']
    scores, cache = affine_forward(out, W, b)
      
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += .5 * self.reg * (
            np.sum([np.sum(self.params['CW' + str(i+1)]**2) for i in range(N)]) + 
            np.sum([np.sum(self.params['AW' + str(i+1)]**2) for i in range(M)]) + 
            np.sum(self.params['W']))
            
    dout, dW, db = affine_backward(dscores, cache)
    grads['b'] = db
    grads['W'] = dW + self.reg * self.params['W']
   
    for j in range(M):
      i = M - j - 1
      if self.use_batchnorm:
        dout, dW, db, dgamma, dbeta= \
                affine_batchnorm_relu_backward(dout, Acache_list[i])
        grads['Agamma' + str(i+1)] = dgamma
        grads['Abeta' + str(i+1)] = dbeta

      else:
        dout, dW, db= \
                affine_relu_backward(dout, Acache_list[i])
      grads['AW' + str(i+1)] = dW + self.reg * self.params['AW'+str(i+1)]
      grads['Ab' + str(i+1)] = db
               
        
    for j in range(N):
      i = N - j - 1
      if self.use_batchnorm:
        dout, dW, db, dgamma, dbeta= \
                conv_batchnorm_relu_pool_backward(dout, Ccache_list[i])
        grads['Cgamma' + str(i+1)] = dgamma
        grads['Cbeta' + str(i+1)] = dbeta

      else:
        dout, dW, db= \
                conv_relu_pool_backward(dout, Ccache_list[i])
      grads['CW' + str(i+1)] = dW + self.reg * self.params['CW'+str(i+1)]
      grads['Cb' + str(i+1)] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
