"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import csv
from tabulate import tabulate

# data I/O
file = csv.reader(open('cyclic.csv','r'))
features = []
for row in file:
  featuresRow = []
  for element in row:
    featuresRow.append(float(element))
  features.append(featuresRow)

amountOfFrames = len(features)
amountOfSamples = len(features[0])
  
print 'data has %d frames, %d samples.' % (amountOfFrames, amountOfSamples)

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, amountOfSamples)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(amountOfSamples, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((amountOfSamples, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are arrays that contain 25 frames.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps, diff = {}, {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.array([inputs[t]]).T
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden layer activations
    ys[t] = np.dot(Why, hs[t]) + by # predictions
    #ys[t][ys[t] < 0] = 0 # predictions must be > 0
    diff[t] = np.subtract(np.array([targets[t]]).T, ys[t])
    loss += np.dot(diff[t].T,diff[t])[0][0]

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = ys[t] - np.array([targets[t]]).T
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed, n):
  x = np.array([seed]).T
  samples = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    samples.append(y.T[0])
    x = y
  print tabulate(np.round(samples,4))

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/amountOfSamples)*seq_length # loss at iteration 0
hprev = np.zeros((hidden_size,1)) # reset RNN memory

def humanizeLargeArray(A):
  smallerA = []
  for i in xrange(0,10):
    smallerA.append(A[i][0:10])
  return smallerA

while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= amountOfFrames or n == 0: 
    p = 0 # go from start of data
    hprev = np.zeros((hidden_size,1)) # reset RNN memory

  inputs = features[p:p+seq_length]
  targets = features[p+1:p+seq_length+1]

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: 
    print 'iter %d, loss: %f, pointer: %d' % (n, smooth_loss, p) # print progress
    sample(hprev,features[p],5)

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
