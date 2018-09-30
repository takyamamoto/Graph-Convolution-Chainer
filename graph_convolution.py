# -*- coding: utf-8 -*-

from chainer import Parameter
from chainer import initializers
from chainer import link
import chainer.functions as F

class GraphConvolution(link.Link):
    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super(GraphConvolution, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            W_initializer = initializers.HeUniform()
            self.W = Parameter(W_initializer, (in_size, out_size))
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = Parameter(bias_initializer, out_size)
        
    def __call__(self, x, adj):
        support = F.matmul(x, self.W)
        output = F.sparse_matmul(adj, support)
        
        if self.b is not None:
            output += self.b
            
        return output