# -*- coding: utf-8 -*-

import argparse

import chainer
from chainer import training
from chainer.training import extensions
from chainer import iterators, optimizers, serializers
import numpy as np

import network

from utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1)
parser.add_argument('--model', '-m', type=str, default=None)
parser.add_argument('--epoch', '-e', type=int, default=200)
parser.add_argument('--lr', '-l', type=float, default=0.01)
parser.add_argument('--noplot', dest='plot', action='store_false',
                    help='Disable PlotReport extension')
args = parser.parse_args()

print("Loading datas")
# Get data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Normalize X
features /= features.sum(1).reshape(-1, 1)
labels = np.expand_dims(labels, 1)

# Inputs
inputs = np.concatenate((features, labels), axis=1)

train_iter = iterators.SerialIterator(inputs, batch_size=adj.shape[0], shuffle=False)
test_iter = chainer.iterators.SerialIterator(inputs, batch_size=adj.shape[0], repeat=False, shuffle=False)

# Set up a neural network to train.
print("Building model")
model = network.GraphConvolutionalNetwork(1433, 16, 7, adj, idx_train, idx_val)

if args.gpu >= 0:
    # Make a specified GPU current
    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()  # Copy the model to the GPU

optimizer = optimizers.Adam(alpha=args.lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

if args.model != None:
    print( "loading model from " + args.model)
    serializers.load_npz(args.model, model)

updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.LogReport())

# Save two plot images to the result dir
if args.plot and extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'), trigger=(10, 'epoch'))
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'), trigger=(10, 'epoch'))

trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

trainer.extend(extensions.ProgressBar())

# Train
trainer.run()

# Save results
print("Optimization Finished!")
modelname = "./results/model"
print( "Saving model to " + modelname)
serializers.save_npz(modelname, model)

# Test
model = network.GraphConvolutionalNetwork(1433, 16, 7, adj, None, idx_test, True)
serializers.load_npz("./results/model", model)
with chainer.using_config('train', False):
    loss_test, acc_test = model(inputs)
print("Test set results:\n",
      "loss =", loss_test.data,
      "\n accuracy =", acc_test.data)
