# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:06:04 2018

@author: user
"""

import numpy as np
import chainer
from chainer import Variable
from chainer import variable
from chainer import reporter
from chainer import initializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

#https://arxiv.org/pdf/1609.02907.pdf
class GraphConv(Chain):
    def __init__(self, inp = 256, out =256):
        super(GraphConv, self).__init__(
            #  入力　出力　フィルタサイズ
            # Weight
            W = L.Linear(inp, out)
        )

        self.inp = inp
        self.out = out
        
        with self.init_scope():
            W_initializer = initializers.Zero()
            self.W = variable.Parameter(W_initializer)
            
    def __call__(self, X, A):
        # H = Activate(output)
        n = A.shape[0]
        
        tilda_A = A + np.eye(n)
        
        tilda_D = np.diag(np.sum(tilda_A, axis=0))
        D = np.diag(np.power(tilda_D,-0.5))
        
        hat_A = D @ tilda_A @ D 
        
        output = hat_A @ self.W(X)
        
        return output

# mainのmodel
class MnistNetwork(Chain):
    def __init__(self, sz=[256, 128, 128], n=256, directory=None):
        super(MnistNetwork, self).__init__(
            e1 = ConvLSTM(n, sz[0], 5),
            e2 = ConvLSTM(sz[0], sz[1], 5),
            e3 = ConvLSTM(sz[1], sz[2], 5),
            p1 = ConvLSTM(n, sz[0], 5),
            p2 = ConvLSTM(sz[0], sz[1], 5),
            p3 = ConvLSTM(sz[1], sz[2], 5),
            last = L.Convolution2D(sum(sz), n, 1)
        )

        self.n = n
        self.directory = directory
        #self.count = 0 #countor
    
        
    def __call__(self, x, t):
        self.e1.reset_state()
        self.e2.reset_state()
        self.e3.reset_state()
        self.count = 0

        #print(self.n)
        We = self.xp.array([[i == j for i in range(self.n)] for j in range(self.n)], dtype=self.xp.float32)
        #print(We.shape) (2,2) <- 256 x 256　のTrue対角行列？
        # trainの方でn=2と設定していただけ
        #print(x.shape[0])
        #print(x.shape[1]) #3 フレーム数
        for i in range(x.shape[1]):
            # x(入力Seq)のフレームを取り出す？
            #print(x.shape) 16 x 3 x 64 x 64
            xi = F.embed_id(x[:, i, :, :], We)
            #print(xi.shape) #(16,64,64,2)
            xi = F.transpose(xi, (0, 3, 1, 2))
            #print(xi.shape) (16, 2, 64, 64)
            
            h1 = self.e1(xi)
            h2 = self.e2(h1)
            self.e3(h2)
            #self.count += 1
            #print(self.count) 1,2,3がループ batch=16でも1batchごとにリセット

        self.p1.reset_state(self.e1.pc, self.e1.ph)
        self.p2.reset_state(self.e2.pc, self.e2.ph)
        self.p3.reset_state(self.e3.pc, self.e3.ph)

        loss = None
        
        xs = x.shape
        for i in range(t.shape[1]):
            #print(xs) #(16,2,64,64) batch x frame x 64 x 64
            # moving mnistの1frameは64x64
            #入ってくるデータはbatchごと
            
            # h1にいれるデータは0配列
            h1 = self.p1(Variable(self.xp.zeros((xs[0], self.n, xs[2], xs[3]), dtype=self.xp.float32)))
            h2 = self.p2(h1)
            h3 = self.p3(h2)

            h = F.concat((h1, h2, h3))
            ans = self.last(h) #output lastは最後のconv
            
            # batch x frame x 32 x 32
            cur_loss = F.softmax_cross_entropy(ans, t[:, i, :, :])
            #loss * const で誤差はconst倍になる
            # 割引率をかける
            #cur_loss = (self.gamma**(t.shape[1]-i))*F.softmax_cross_entropy(ans, t[:, i, :, :])
            loss = cur_loss if loss is None else loss + cur_loss
            
        reporter.report({'loss': loss}, self)
        
        return loss