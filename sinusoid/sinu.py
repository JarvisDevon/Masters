import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time
import jax
from jax.api import jit, grad
from jax.scipy.special import logsumexp
from jax.flatten_util import ravel_pytree
from jax import lax
import jax.numpy as jnp
import datasets
import wandb
import log_handler

# Functions which help with basic use of all architectures
def init_random_params(scale, layer_sizes, seed):
  np.random.seed(seed)
  return [(scale * np.random.randn(m, n), scale * np.random.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

# Computes the ReLU activation function
def relu(x):
    return jnp.maximum(x, 0)

# Computes the sigmoid activation function
def sigmoid(x):
    return jnp.where(x >= 0, 1/(1+jnp.exp(-x)),  jnp.exp(x)/(1+jnp.exp(x)))

@jit
def predict(params, inputs):
  for W, b in params[0:len(params)-2]:
      outputs = jnp.dot(inputs, W) + b
      inputs = relu(outputs) #jnp.tanh(outputs) #sigmoid(outputs)
  for W, b in params[len(params)-2:]:
      outputs = jnp.dot(inputs, W) + b
      inputs = outputs
  return outputs

@jit
def accuracy(params, batch):
  inputs, targets = batch
  net_out = predict(params, inputs)
  return (1/inputs.shape[0])*jnp.sum(jnp.power((net_out - targets),2))

# Functions which computes the Power Method using the Efficient Hessian Multiplication
@jit
def params_matmul(params, batch, const_vector):
  grads = grad(loss)(params, batch)
  grads_flat, _ = ravel_pytree(grads)
  return jnp.dot(const_vector.T, grads_flat)

gpus = jax.devices('gpu')
cpus = jax.devices('cpu')
jit_QR = jax.jit(jnp.linalg.qr)#, device=gpus[1])
jit_error = jax.jit(lambda Q, Q_prev: ((jnp.abs(Q) - jnp.abs(Q_prev)) ** 2).sum())#, device=gpus[1])

def param_simultaneous_power_iteration(params, k, batch, init_Q):
    Q, _ = jit_QR(init_Q) #jnp.linalg.qr(init_Q)
    Q_prev = Q
    for i in range(100):
        Q, R = update_power_iteration(Q, params, batch)
        # can use other stopping criteria as well 
        err = jit_error(Q, Q_prev)
        if err < 0.001: 
            break
    return jnp.abs(jnp.diag(R)), Q

@jit
def update_power_iteration(Q, params, batch):
    Z = jnp.zeros(Q.shape)
    for j in range(Q.shape[1]):
        update_vec = grad(params_matmul)(params, batch, Q[:,j])
        Z = jax.ops.index_update(Z, jax.ops.index[:,j], ravel_pytree(update_vec)[0])
    Q, R = jnp.linalg.qr(Z)
    return Q,R

# Functions for the different possible pieces of a loss function
@jit
def loss(params, batch):
  inputs, targets = batch
  net_out = predict(params, inputs)
  return (1/inputs.shape[0])*jnp.sum(jnp.power((net_out - targets),2))

@jit
def mahalanobis_dist(params, batch, params_constant_ravel):
  inputs, targets = batch
  grads = grad(loss)(params, batch)
  grads_flat, _ = ravel_pytree(grads)
  return jnp.dot(params_constant_ravel.T, grads_flat)

#@jit
def param_volume(params, batch, init_Q):
  eig_vals, _ = param_simultaneous_power_iteration(params, wandb.config.num_eigs_keep, batch, init_Q)
  return(-jnp.log(jnp.prod(eig_vals)))

@jit
def l2_regularizer(params):
  flat_params, unflatten = ravel_pytree(params)
  return jnp.dot(flat_params.T, flat_params)

# Run experiments
if __name__ == "__main__":
  wandb.init()

  # Function which updates model parameters without regularization
  @jit
  def update_standard(params, batch, epoch, init_Q):
    grads = grad(loss)(params, batch)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

  # Function which updates model parameters with mahalanobis dist regularizer
  @jit
  def update_mahal(params, batch, epoch, init_Q):
    const_flat_params, unflattener = ravel_pytree(params)
    grads = grad(loss)(params, batch)
    mahala_grads = grad(mahalanobis_dist)(params, batch, const_flat_params)
    return [(w-step_size*dw - wandb.config.reg_step*jnp.multiply(jnp.abs(dw_mah),jnp.sign(w)),\
             b-step_size*db - wandb.config.reg_step*jnp.multiply(jnp.abs(db_mah),jnp.sign(b)))\
             for (w, b),(dw, db),(dw_mah, db_mah) in zip(params,grads,mahala_grads)]

  # Function which updates model parameters with full regularizer
  @jit
  def update_vol(params, batch, epoch, init_Q):
    grads = grad(loss)(params, batch)
    volume_grads = grad(param_volume)(params, batch, init_Q)
    return [(w - step_size*dw - wandb.config.vol_step*dw_vol, b - step_size*db - wandb.config.vol_step*db_vol)\
           for (w, b),(dw, db),(dw_vol, db_vol) in zip(params,grads,volume_grads)]

  # Function which updates model parameters with L2 regularization
  @jit
  def update_l2(params, batch, epoch, init_Q):
    grads = grad(loss)(params, batch)
    reg_grads = grad(l2_regularizer)(params)
    return [(w - step_size*dw - wandb.config.l2_step*dw_reg, b - step_size*db - wandb.config.l2_step*db_reg) \
           for (w, b),(dw, db),(dw_reg, db_reg) in zip(params,grads,reg_grads)]

  @jit
  def update_mahal_vol_jax_bit(params, batch):
    const_flat_params, unflattener = ravel_pytree(params)
    grads = grad(loss)(params, batch)
    mahala_grads = grad(mahalanobis_dist)(params, batch, const_flat_params)
    return grads, mahala_grads

  # Function which updates model parameters with full regularizer
  #@jit
  def update_mahal_vol(params, batch, epoch, init_Q):
    grads, mahala_grads = update_mahal_vol_jax_bit(params, batch)
    volume_grads = grad(param_volume)(params, batch, init_Q)
    return [(w - step_size*dw - wandb.config.reg_step*dw_mah - wandb.config.vol_step*dw_vol, \
             b - step_size*db - wandb.config.reg_step*db_mah - wandb.config.vol_step*db_vol) \
           for (w, b),(dw, db),(dw_mah, db_mah),(dw_vol, db_vol) in zip(params,grads,mahala_grads,volume_grads)]
 
  # Function which updates model parameters with mahalanobis dist regularizer
  @jit
  def update_mahal_l2(params, batch, epoch, init_Q):
    const_flat_params, unflattener = ravel_pytree(params)
    grads = grad(loss)(params, batch)
    mahala_grads = grad(mahalanobis_dist)(params, batch, const_flat_params)
    reg_grads = grad(l2_regularizer)(params)
    return [(w-step_size*dw - wandb.config.l2_step*dw_reg - wandb.config.reg_step*jnp.multiply(jnp.abs(dw_mah),jnp.sign(w)),\
             b-step_size*db - wandb.config.l2_step*db_reg - wandb.config.reg_step*jnp.multiply(jnp.abs(db_mah),jnp.sign(b)))\
             for (w, b),(dw, db),(dw_mah, db_mah),(dw_reg, db_reg) in zip(params,grads,mahala_grads,reg_grads)]

  @jit
  def update_Selective(params, batch, epoch, init_Q):
    const_flat_params, unflattener = ravel_pytree(params)
    important_params = grad(mahalanobis_dist)(params, batch, const_flat_params)
    grads = grad(loss)(params, batch)
    reg_grads = grad(l2_regularizer)(params) 
    return [(w-step_size*dw-wandb.config.selective_l2_step*jnp.where(jnp.abs(dw_choose)>0.005*jnp.max(dw_choose),0.0,dw_reg),\
             b-step_size*db-wandb.config.selective_l2_step*jnp.where(jnp.abs(db_choose)>0.005*jnp.max(db_choose),0.0,db_reg))\
             for (w, b),(dw, db),(dw_reg, db_reg),(dw_choose, db_choose) in zip(params,grads,reg_grads,important_params)]

  # Function which updates model parameters with L2 regularization
  #@jit
  def update_l2_stop(params, batch, epoch, init_Q):
    if epoch < int(0.6*wandb.config.num_epochs):
        return update_l2(params, batch, epoch, init_Q)
    else:
        return update_standard(params, batch, epoch, init_Q)

  #@jit
  def update_l2_stop_selective(params, batch, epoch, init_Q):
    if epoch < int(0.6*wandb.config.num_epochs):
        return update_l2(params, batch, epoch, init_Q)
    else:
        return update_Selective(params, batch, epoch, init_Q)

  # Set hyper-parameters
  wandb.config.layer_sizes = [256, 128, 56, 1]
  wandb.config.param_scale_regime_1 = 0.01
  wandb.config.param_scale_regime_2 = 0.02
  wandb.config.param_scale_regime_3 = 0.2 
  wandb.config.step_size_regime_2 = 2e-4 
  wandb.config.step_size_regime_3 = 6e-5
  wandb.config.reg_step = 2e-6 
  wandb.config.vol_step = 1e-5 
  wandb.config.l2_step = 1e-5
  wandb.config.selective_l2_step = 0.001
  wandb.config.num_epochs = 1000
  wandb.config.batch_size = 32
  wandb.config.num_eigs_keep = 3
  wandb.config.num_trainings = 30
  wandb.config.data_shape = (2000,256)
  wandb.config.inputs_scale = 2.0 
  wandb.config.gen_params_scale = 1.0 
  wandb.config.outputs_scale = 1.0
  wandb.config.training_names = ['SGD Small Regime','SGD Large Regime','Metric Small Regime','Metric Large Regime',\
                                 'L2 Large Regime','Selective L2 Large Regime','L2 Stopped Large Regime',\
                                 'Selective L2 Stopped Large Regime']
  wandb.config.updater_index = [0, 0, 1, 1, 3, 6, 7, 8]
  wandb.config.init_index = [1, 2, 1, 2, 2, 2, 2, 2]
  wandb.config.init_type = [wandb.config.param_scale_regime_1, wandb.config.param_scale_regime_2, wandb.config.param_scale_regime_3]
  wandb.config.init_seeds = np.random.choice(100000000, wandb.config.num_trainings, replace=False)

  updaters = [update_standard, update_mahal, update_vol, update_l2,\
              update_mahal_vol, update_mahal_l2,\
              update_Selective, update_l2_stop, update_l2_stop_selective]

  # Loading the dataset
  train_images, train_labels, test_images, test_labels, num_batches, batches = \
                                  datasets.gen_sin_data(wandb.config.data_shape, wandb.config.inputs_scale, \
                                                        wandb.config.gen_params_scale, wandb.config.outputs_scale, wandb.config.batch_size) 

  # Create log file and arrays to track the repeated errors
  logs = log_handler.init_logs(len(wandb.config.training_names), wandb.config.num_epochs, wandb.config.num_eigs_keep)

  print("Set of Updates: ", wandb.config.training_names)

  num_params = jnp.sum([i*j + j for i,j in zip(wandb.config.layer_sizes[:-1], wandb.config.layer_sizes[1:])])
  track_Q = np.random.rand(num_params, wandb.config.num_eigs_keep)*100

  for train_index in range(len(wandb.config.training_names)):
      for seed in wandb.config.init_seeds:
          print("############################# New "+wandb.config.training_names[train_index]+" Training ###############################")
          params = init_random_params(wandb.config.init_type[wandb.config.init_index[train_index]], wandb.config.layer_sizes, seed)
          updater = updaters[wandb.config.updater_index[train_index]]
          if wandb.config.init_index[train_index] == 1:
              step_size = wandb.config.step_size_regime_2
          elif wandb.config.init_index[train_index] == 2:
              step_size = wandb.config.step_size_regime_3
          run_logs =  log_handler.init_run_logs() # training, test, time logs, weight norm
          for epoch in range(wandb.config.num_epochs):
              start_time = time.time()
              for _ in range(num_batches):
                params = updater(params, next(batches), epoch, track_Q)
              epoch_time = time.time() - start_time
              norm = np.sum([np.linalg.norm(param_set[0])+np.linalg.norm(param_set[1]) for param_set in params])

              train_acc = accuracy(params, (train_images, train_labels))
              test_acc = accuracy(params, (test_images, test_labels))
              print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
              print("Training set accuracy {}".format(train_acc))
              print("Test set accuracy {}".format(test_acc))
              wandb.log({"Training Type": wandb.config.training_names[train_index], "epoch": epoch, "epoch_time": epoch_time,\
                         wandb.config.training_names[train_index]+" Train Accuracy": float(train_acc),\
                         wandb.config.training_names[train_index]+" Test Accuracy": float(test_acc)})

              run_logs[0] = np.append(run_logs[0], train_acc)
              run_logs[1] = np.append(run_logs[1], test_acc)
              run_logs[2] = np.append(run_logs[2], epoch_time)
              run_logs[3] = np.append(run_logs[3], norm)

          logs[0 + (train_index*5)] = np.vstack([logs[0 + (train_index*5)], run_logs[0].reshape(1,wandb.config.num_epochs)])
          logs[1 + (train_index*5)] = np.vstack([logs[1 + (train_index*5)], run_logs[1].reshape(1,wandb.config.num_epochs)])
          logs[2 + (train_index*5)] = np.vstack([logs[2 + (train_index*5)], run_logs[2].reshape(1,wandb.config.num_epochs)])
          converge_eig_vals, track_Q = param_simultaneous_power_iteration(params, wandb.config.num_eigs_keep,\
                                                                         (train_images, train_labels), track_Q)
          logs[3 + (train_index*5)] = np.vstack([logs[3 + (train_index*5)], converge_eig_vals])
          logs[4 + (train_index*5)] = np.vstack([logs[4 + (train_index*5)], run_logs[3].reshape(1,wandb.config.num_epochs)])

  # Remove place holder first line of zeros used to just define the shape of the arrays to vstack with initially
  all_errors=log_handler.remove_place_holders(logs) 

  # Calculate the means of the train and test errors for all types of training, as well as for the difference in errors of the
  # different trainings to the full regularizer (excluding full reg with itself)
  all_means = log_handler.calc_all_means(all_errors, wandb.config.training_names)

  # Calculate the standard deviations of the train and test errors for all types of training, and for the difference in errors of the
  # different trainings to the full regularizer (excluding full reg with itself)
  all_std = log_handler.calc_all_std(all_errors, wandb.config.training_names)

  # Plots the mean training and test error/accuracy (for regression/classification respectively) for each training type using the
  # mean values calculated above with 2*standard deviation error bounds
  log_handler.create_all_plots(all_means, all_std, wandb.config.training_names, "Error", "Sinusoid Regression", True)
