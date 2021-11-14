# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Datasets used in examples."""


import array
import gzip
import os
from os import path
import struct
import urllib.request

import numpy as np
import jax.numpy as jnp
import numpy.random as npr
import pickle
import os

_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(_DATA):
    os.makedirs(_DATA)
  out_file = path.join(_DATA, filename)
  if not path.isfile(out_file):
    urllib.request.urlretrieve(url, out_file)
    print("downloaded {} to {}".format(url, _DATA))


def _partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def sigmoid(x):
  return np.where(x >= 0, 1/(1+np.exp(-x)),  np.exp(x)/(1+np.exp(x)))

def mnist_raw():
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(base_url + filename, filename)

  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

  return train_images, train_labels, test_images, test_labels


def mnist(batch_size, permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten(train_images) / np.float32(255.)
  test_images = _partial_flatten(test_images) / np.float32(255.)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  return train_images, train_labels, test_images, test_labels, num_batches, data_stream()

def mnist_disp(batch_size, permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten(train_images) / np.float32(255.)
  test_images = _partial_flatten(test_images) / np.float32(255.)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)
  samp = np.random.randint(0, 2, train_images.shape)
  train_images[np.where(samp)] = train_images[np.where(samp)]*100000.0

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  return train_images, train_labels, test_images, test_labels, num_batches, data_stream()


def mnist_regression(batch_size, permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten(train_images) / np.float32(255.)
  test_images = _partial_flatten(test_images) / np.float32(255.)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)
  train_labels = np.where(train_labels==1, 0.9, -0.1)
  test_labels = np.where(test_labels==1, 0.9, -0.1)

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]
 
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  return train_images, train_labels, test_images, test_labels, num_batches, data_stream()

def gen_sin_data(input_dims, inputs_scale, params_scale, outputs_scale, batch_size, noise_var = 1.0):
  print("Start load", flush=True)
  inputs = np.random.rand(*input_dims)*inputs_scale
  gen_params = np.random.rand(input_dims[1], 1)*params_scale
  bias = np.random.rand(input_dims[1], 1)*params_scale
  labels = outputs_scale*jnp.sin(jnp.dot(inputs, gen_params)) + outputs_scale*jnp.dot(inputs, bias)
  labels = labels + np.random.normal(0.0, noise_var, labels.shape)
  data_split = int(0.8*input_dims[0])
  print("Gen data with variance: ", np.var(labels))

  labels = labels - np.mean(labels) # 0 mean the data
  train_images, train_labels, test_images, test_labels = inputs[:data_split], labels[:data_split], inputs[data_split:], labels[data_split:]

  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  return train_images, train_labels, test_images, test_labels, num_batches, data_stream()

def gen_synth_data(input_dims, test_dims, odd_scale, even_scale, noise_scale, batch_size):
  # Randomly construct a ground truth network
  true_net_size = np.random.randint(5,10)*2 # times by two to make sure we get an even number
  true_layers = np.append(np.random.randint(5,100, true_net_size), 1)
  true_layers.sort()
  true_layers = true_layers[::-1]
  true_layers[0] = input_dims[1]
  print(true_layers)
  position = np.random.uniform(-0.5, 0.5)
  true_model = []
  gen_layers = [ [( np.random.normal(position, odd_scale,size=(m,n)),\
                np.random.normal(position, odd_scale, size=(n,)) ),\
                ( np.random.normal(position, even_scale,size=(p,q)),\
                np.random.normal(position, even_scale, size=(q,)) )] for m,n,p,q in\
                zip(true_layers[:-2:2], true_layers[1:len(true_layers)-1:2], true_layers[1:len(true_layers)-1:2], true_layers[2::2])]
  for layer_pair in gen_layers:
    true_model.extend(layer_pair)
  
  # Gen training data and labels using network
  train_data = np.random.uniform(0.0, 1.0, input_dims)
  inputs = np.copy(train_data)
  for W, b in true_model:
    train_labels = np.dot(inputs, W) + b
    inputs = sigmoid(train_labels)
  train_labels = train_labels + np.random.normal(0.0, noise_scale, train_labels.shape)
  train_labels = train_labels - np.mean(train_labels)

  # Gen test data and labels using network
  test_data = np.random.uniform(1.0, 2.0, test_dims)
  test_inputs = np.copy(test_data)
  for W, b in true_model:
    test_labels = np.dot(test_inputs, W) + b
    test_inputs = sigmoid(test_labels)
  test_labels = test_labels - np.mean(train_labels) #use mean of train labels for consistency

  # Calculate batches
  num_train = train_data.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  # Data loader
  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_data[batch_idx], train_labels[batch_idx]

  return train_data, train_labels, test_data, test_labels, num_batches, data_stream()


def cifar10_loaders(batch_size):
  num_train = 10000 # size of a normal batch
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def load_cfar10_batch():
    rng = npr.RandomState()
    while True:
        batch_id = npr.randint(1,5)#please set back to 6 when including test set
        #print("Loading file: ", batch_id)
        with open('cifar-10-batches-py/data_batch_' + str(batch_id), mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')
        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(2, 3, 1,0).astype(np.float64)/255.0 #(0, 2, 3, 1)
        labels = _one_hot(np.array(batch['labels']), 10).astype(np.float16)
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            #print("Returning batch: ", i)
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield features[:,:,:,batch_idx], labels[batch_idx]

  def load_cfar10_batch_test():
    while True:
        with open('cifar-10-batches-py/test_batch', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')
        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(2, 3, 1, 0).astype(np.float64)/255.0 #(0, 2, 3, 1)
        labels = _one_hot(np.array(batch['labels']), 10).astype(np.float16)
        yield features, labels

  return num_batches, load_cfar10_batch(), load_cfar10_batch_test()
