import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time
from jax.api import jit, grad
from jax import hessian as hess
from jax.scipy.special import logsumexp
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import datasets
import wandb
import log_handler

# Functions which help with basic use of all architectures
def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

# Computes the sigmoid activation function
def sigmoid(x):
    return jnp.where(x >= 0, 1/(1+jnp.exp(-x)),  jnp.exp(x)/(1+jnp.exp(x)))

def predict(params, inputs):
  for W, b in params[0:len(params)-2]:
      outputs = jnp.dot(inputs, W) + b 
      inputs = sigmoid(outputs)
  for W, b in params[len(params)-2:]:
      outputs = jnp.dot(inputs, W) + b 
      inputs = outputs
  return outputs

def accuracy(params, batch):
  inputs, targets = batch
  net_out = predict(params, inputs)
  return (1/inputs.shape[0])*jnp.sum(jnp.power((net_out - targets),2))

# Functions for the different possible pieces of a loss function
def loss(params, batch):
  inputs, targets = batch
  net_out = predict(params, inputs)
  return (1/inputs.shape[0])*jnp.sum(jnp.power((net_out - targets),2))

def loss_flat(flat_params, batch, unflattener):
  unflat_params = unflattener(flat_params)
  return loss(unflat_params, batch)

#def param_volume(params, batch):
#  inputs, targets = batch
#  grads = grad(loss)(params, batch)
#  param_grads, _ = ravel_pytree(grads)
#  param_grads = param_grads.reshape(param_grads.shape[0],1)
#  eig_vals, _ = get_eigs.param_simultaneous_power_iteration(param_grads, wandb.config.num_eigs_keep)
#  return(-jnp.log(jnp.prod(eig_vals)))

@jit
def params_matmul(params, batch, const_vector):
  grads = grad(loss)(params, batch)
  grads_flat, _ = ravel_pytree(grads)
  return jnp.dot(const_vector.T, grads_flat)

def simultaneous_power_iteration(A, k):
    n, m = A.shape
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q

    for i in range(1000):
        Z = jnp.dot(A, Q)
        Q, R = jnp.linalg.qr(Z)

        # can use other stopping criteria as well 
        err = ((Q - Q_prev) ** 2).sum()
        Q_prev = Q
        if err < 1e-3:
            break

    return np.diag(R), Q

def param_simultaneous_power_iteration(params, k, batch):
    ravel_params, unflattener = ravel_pytree(params)
    n = ravel_params.shape[0]
    m = ravel_params.shape[0]
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q
    for i in range(1000):
        Z = np.zeros(Q.shape)
        for j in range(Q.shape[1]):
            update_vec = grad(params_matmul)(params, batch, Q[:,j])
            Z[:,j] = ravel_pytree(update_vec)[0]
        Q, R = jnp.linalg.qr(Z)
        # can use other stopping criteria as well 
        err = ((Q - Q_prev) ** 2).sum()
        Q_prev = Q
        if err < 1e-3:
            break

    return jnp.abs(jnp.diag(R)), Q


# Run experiments
if __name__ == "__main__":
  wandb.init()

  # Set hyper-parameters
  wandb.config.layer_sizes = [50, 5, 2, 1]
  wandb.config.param_scale = 0.08 
  wandb.config.step_size = 1e-3
  wandb.config.num_epochs = 200 
  wandb.config.batch_size = 200
  wandb.config.num_eigs_keep = 3 
  wandb.config.data_shape = (200,50)
  wandb.config.inputs_scale = 2.0 
  wandb.config.gen_params_scale = 1.0 
  wandb.config.outputs_scale = 1.0

  # Loading the dataset
  train_images, train_labels, test_images, test_labels, num_batches, batches = \
                                    datasets.gen_sin_data(wandb.config.data_shape, wandb.config.inputs_scale, \
                                                        wandb.config.gen_params_scale, wandb.config.outputs_scale, wandb.config.batch_size)

  # Function which updates model parameters without regularization
  @jit
  def update_standard(params, batch):
    grads = grad(loss)(params, batch)
    return [(w - wandb.config.step_size * dw, b - wandb.config.step_size * db) for (w, b), (dw, db) in zip(params, grads)]
 
  @jit
  def fast_hess(params, batch, params_constant_ravel):
    grads = grad(loss)(params, batch)
    grads_flat, _ = ravel_pytree(grads)
    return jnp.dot(params_constant_ravel.T, grads_flat)

  print("############################# Standard Update Test ###############################")
  params = init_random_params(wandb.config.param_scale, wandb.config.layer_sizes)
  for epoch in range(wandb.config.num_epochs):
    for _ in range(num_batches):
      params = update_standard(params, next(batches))

    # Checking Multiplication
    model_params_flat, unflattener = ravel_pytree(params)
    check_hess_product = grad(fast_hess)(params, (train_images, train_labels), model_params_flat)
    hessian = hess(loss_flat)(model_params_flat, (train_images, train_labels), unflattener)
    truth_hess_product = jnp.dot(model_params_flat.T, hessian)
    max_difference_mult = jnp.max(jnp.abs(ravel_pytree(check_hess_product)[0] - truth_hess_product))

    # Checking Power Method
    true_eig_vals = np.linalg.eigvals(hessian)
    true_eig_vals = np.sort(true_eig_vals)[::-1]
    true_eig_vals = true_eig_vals[:wandb.config.num_eigs_keep]
    check_eig_vals, _ = param_simultaneous_power_iteration(params, wandb.config.num_eigs_keep, (train_images, train_labels))
    max_difference_power = jnp.max(jnp.abs(check_eig_vals - true_eig_vals))
    comp_power_method, _ = simultaneous_power_iteration(hessian, wandb.config.num_eigs_keep)

    # Tracking Accuracies may be helpful
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    print("Maximum multiplication difference {}".format(max_difference_mult))
    print("Maximum power method eig-value difference {}".format(max_difference_power))
    print("True Eigenvalues: {}".format(true_eig_vals))
    print("Found Eigenvalues: {}".format(check_eig_vals))
    print("Other Power Eigenvalues: {}".format(comp_power_method))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
    wandb.log({"Maximum multiplication difference": float(max_difference_mult),\
               "Maximum power method eig-value difference": float(max_difference_power),
               "Train Accuracy": float(train_acc), "Test Accuracy": float(test_acc)})
