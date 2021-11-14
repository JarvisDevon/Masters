import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time
from functools import partial
import jax
from jax import jit, grad, random
from jax.flatten_util import ravel_pytree
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, Dropout, FanInSum,                                                       
                                   FanOut, Flatten, GeneralConv, Identity,                                                                   
                                   MaxPool, Tanh, Relu, LogSoftmax)
from jax.nn.initializers import variance_scaling, normal, glorot_normal
import jax.numpy as jnp
import datasets
import wandb
import log_handler

def scale_tree_sgd(grad_tree):
    return wandb.config.step_size*grad_tree

def add_two_trees_mahal(grad_tree, reg_tree):
    return wandb.config.step_size*grad_tree + wandb.config.reg_step*reg_tree

def add_two_trees_l2(grad_tree, l2_tree):
    return wandb.config.step_size*grad_tree + wandb.config.l2_step*l2_tree

def add_two_trees_selective_l2(grad_tree, l2_tree, selection_tree, max_tree):
    return wandb.config.step_size*grad_tree + wandb.config.selective_l2_step*l2_tree*(jnp.abs(selection_tree)>0.05*max_tree)

def add_three_trees(tree1, tree2, tree3):
    return tree1 + tree2 + tree3

# Architecture Definition
def ResNet50_small(num_classes):
  return stax.serial(
      GeneralConv(('HWCN', 'OIHW', 'NHWC'),32, (7, 7), (1, 1), 'VALID', W_init = scaled_glorot_small()),Relu,MaxPool((2, 2), strides=(2, 2)),
      Conv(64, (2, 2), W_init = scaled_glorot_small(), padding='VALID'), Relu, 
      Flatten, Dense(500, W_init = scaled_glorot_small()), Relu,
      Dense(100, W_init = scaled_glorot_small()), Relu,
      Dense(50, W_init = scaled_glorot_small()), Relu,
      Dense(20, W_init = scaled_glorot_small()), Relu,
      Dense(num_classes, W_init = scaled_glorot_small()),
      LogSoftmax)

# Architecture Definition
def ResNet50_large(num_classes):
  return stax.serial(
      GeneralConv(('HWCN', 'OIHW', 'NHWC'),32, (7, 7), (1, 1), 'VALID', W_init = scaled_glorot_large()),Relu,MaxPool((2, 2), strides=(2, 2)),
      Conv(64, (2, 2), W_init = scaled_glorot_large(), padding='VALID'), Relu,
      Flatten, Dense(500, W_init = scaled_glorot_large()), Relu,
      Dense(100, W_init = scaled_glorot_large()), Relu,
      Dense(50, W_init = scaled_glorot_large()), Relu,
      Dense(20, W_init = scaled_glorot_large()), Relu,
      Dense(num_classes, W_init = scaled_glorot_large()),
      LogSoftmax)

@jit
def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=-1)
  predicted_class = jnp.argmax(predict_fun(params, inputs), axis=-1)
  return jnp.mean(predicted_class == target_class)

# Functions which computes the Power Method using the Efficient Hessian Multiplication
@jit
def params_matmul(params, batch, const_vector):
  grads = grad(loss)(params, batch)
  grads_flat, _ = ravel_pytree(grads)
  return jnp.dot(const_vector.T, grads_flat)

gpus = jax.devices('gpu')
cpus = jax.devices('cpu')
jit_QR = jax.jit(jnp.linalg.qr, device=gpus[0])
jit_grad = jax.jit(grad(params_matmul), device=gpus[0])
jit_zeros = jax.jit(jnp.zeros, static_argnums=[0], device=gpus[0])
jit_error = jax.jit(lambda Q, Q_prev: ((jnp.abs(Q) - jnp.abs(Q_prev)) ** 2).sum(), device=gpus[0])

# Performs power iteration on the parameters
def param_simultaneous_power_iteration(params, k, batch, init_Q):
    Q, _ = jnp.linalg.qr(init_Q)
    Q_prev = Q
    for i in range(100):
        Q, R = update_power_iteration(Q, params, batch, Q.shape)
        # can use other stopping criteria as well 
        err = jit_error(Q, Q_prev)
        print("Error: ", err) 
        Q_prev = np.copy(Q)
        if err < 0.1: #0.001 #1.0
            break
    return jnp.abs(jnp.diag(R)), Q

def update_power_iteration(Q, params, batch, z_shape):
    Z = jit_zeros(z_shape)
    for j in range(Q.shape[1]):
        update_vec = jit_grad(params, batch, Q[:,j])
        update_vec_ravel = ravel_pytree(update_vec)[0]
        Z = jax.ops.index_update(Z, jax.ops.index[:,j], update_vec_ravel)
    Q, R = jit_QR(Z) #jnp.linalg.qr(Z)
    return Q,R

update_power_iteration = jax.jit(update_power_iteration, static_argnums=[3], device=cpus[0])

# Functions for the different possible pieces of a loss function
@jit
def loss(params, batch):
  inputs, targets = batch
  preds = predict_fun(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

@jit
def mahalanobis_dist(params, batch, params_constant_ravel):
  inputs, targets = batch
  grads = grad(loss)(params, batch)
  grads_flat, _ = ravel_pytree(grads)
  return jnp.dot(params_constant_ravel.T, grads_flat)

#@jit
def param_volume(params, batch):
  eig_vals, _ = param_simultaneous_power_iteration(params, wandb.config.num_eigs_keep, batch)
  return(-jnp.log(jnp.prod(eig_vals)))

@jit
def l2_regularizer(params):
  flat_params, unflatten = ravel_pytree(params)
  return jnp.dot(flat_params.T, flat_params)

# Run experiments
if __name__ == "__main__":
  wandb.init()

  @jit
  def update_standard(opt_state, batch, epoch):#, init_Q):
    params = get_params(opt_state)
    grads = grad(loss)(params, batch)
    full_grads = jax.tree_util.tree_map(scale_tree_sgd, grads)
    return opt_update(epoch, full_grads, opt_state)

  # Function which updates model parameters with mahalanobis dist regularizer
  @jit
  def update_mahal(opt_state, batch, epoch):#, init_Q):
    params = get_params(opt_state)
    const_flat_params, unflattener = ravel_pytree(params)
    grads = grad(loss)(params, batch)
    mahala_grads = grad(mahalanobis_dist)(params, batch, const_flat_params) #wandb.config.reg_step/wandb.config.step_size
    full_grads = jax.tree_util.tree_multimap(add_two_trees_mahal, grads, (mahala_grads))
    return opt_update(epoch, full_grads, opt_state)

  # Function which updates model parameters with full regularizer
  @jit
  def update_vol(opt_state, batch, epoch, init_Q):
    params = get_params(opt_state)
    grads = grad(loss)(params, batch)
    volume_grads = grad(param_volume)(params, batch, init_Q)
    full_grads = [(step_size*dw + wandb.config.vol_step*dw_vol, step_size*db + wandb.config.vol_step*db_vol)\
                  for (dw, db),(dw_vol, db_vol) in zip(grads,volume_grads)]
    return opt_update(i, full_grads, opt_state)

  # Function which updates model parameters with L2 regularization
  @jit
  def update_l2(opt_state, batch, epoch):#, init_Q):
    params = get_params(opt_state)
    grads = grad(loss)(params, batch)
    reg_grads = grad(l2_regularizer)(params)
    full_grads = jax.tree_util.tree_multimap(add_two_trees_l2, grads, (reg_grads))
    return opt_update(epoch, full_grads, opt_state)

  @jit
  def update_mahal_vol_jax_bit(params, batch):
    const_flat_params, unflattener = ravel_pytree(params)
    grads = grad(loss)(params, batch)
    mahala_grads = grad(mahalanobis_dist)(params, batch, const_flat_params)
    return grads, mahala_grads

  # Function which updates model parameters with full regularizer
  #@jit
  def update_mahal_vol(opt_state, batch, epoch, init_Q):
    params = get_params(opt_state)
    grads, mahala_grads = update_mahal_vol_jax_bit(params, batch)
    volume_grads = grad(param_volume)(params, batch, init_Q)
    full_grads = [(step_size*dw + reg_step*dw_mah + vol_step*dw_vol, \
                   step_size*db + reg_step*db_mah + vol_step*db_vol) \
                   for (dw, db),(dw_mah, db_mah),(dw_vol, db_vol) in zip(grads,mahala_grads,volume_grads)]
    return opt_update(i, full_grads, opt_state)

  # Function which updates model parameters with mahalanobis dist regularizer
  @jit
  def update_mahal_l2(opt_state, batch, epoch):#, init_Q):
    params = get_params(opt_state)
    const_flat_params, unflattener = ravel_pytree(params)
    grads = grad(loss)(params, batch)
    mahala_grads = grad(mahalanobis_dist)(params, batch, const_flat_params)
    reg_grads = grad(l2_regularizer)(params)
    full_grads = [(step_size*dw+wandb.config.l2_step*dw_reg+wandb.config.reg_step*jnp.multiply(jnp.abs(dw_mah),jnp.sign(w)),\
                   step_size*db+wandb.config.l2_step*db_reg+wandb.config.reg_step*jnp.multiply(jnp.abs(db_mah),jnp.sign(b)))\
                   for (dw, db),(dw_mah, db_mah),(dw_reg, db_reg) in zip(grads,mahala_grads,reg_grads)]
    return opt_update(i, full_grads, opt_state)

  @jit
  def update_Selective(opt_state, batch, epoch):#, init_Q):
    params = get_params(opt_state)
    const_flat_params, unflattener = ravel_pytree(params)
    important_params = grad(mahalanobis_dist)(params, batch, const_flat_params)
    #rint(jax.tree_util.tree_structure(important_params))
    max_params = important_params.copy()
    for i in [0,3,6,8,10,12,14]:
        max_params[i] = tuple([jnp.ones(max_params[i][0].shape)*jnp.max(max_params[i][0]),\
                              jnp.ones(max_params[i][1].shape)*jnp.max(max_params[i][1])])
    grads = grad(loss)(params, batch)
    reg_grads = grad(l2_regularizer)(params)
    full_grads = jax.tree_util.tree_multimap(add_two_trees_selective_l2, grads, reg_grads, important_params, max_params)
    return opt_update(epoch, full_grads, opt_state)

  # Function which updates model parameters with L2 regularization
  #@jit
  def update_l2_stop(opt_state, batch, epoch):
    if epoch < int(0.7*num_epochs):
        return update_l2(opt_state, batch, epoch)
    else:
        return update_standard(opt_state, batch, epoch)

  #@jit
  def update_l2_stop_selective(opt_state, batch, epoch):
    if epoch < int(0.7*num_epochs):#0.7
        return update_l2(opt_state, batch, epoch)
    else:
        return update_Selective(opt_state, batch, epoch)

  # Set hyper-parameters
  wandb.config.param_scale_regime_1 = 0.01
  wandb.config.param_scale_regime_2 = 0.2
  wandb.config.param_scale_regime_3 = 5.0
  wandb.config.step_size = 0.005 
  wandb.config.reg_step = 0.00005 
  wandb.config.l2_step = 0.00005 
  wandb.config.selective_l2_step = 0.00005
  wandb.config.num_epochs = 1000
  wandb.config.batch_size = 32 
  wandb.config.num_eigs_keep = 10 
  wandb.config.num_classes = 10
  wandb.config.input_shape = (32, 32, 3, wandb.config.batch_size)
  wandb.config.num_trainings = 3 
  wandb.config.training_names = ['SGD Small Regime','SGD Large Regime','Metric Small Regime','Metric Large Regime',\
                                 'L2 Large Regime','Selective L2 Large Regime','L2 Stopped Large Regime',\
                                 'Selective L2 Stopped Large Regime']
  wandb.config.updater_index = [6, 0, 1, 1, 3, 6, 7, 8]
  wandb.config.init_index = [2, 2, 1, 2, 2, 2, 2, 2]
  wandb.config.init_type = [wandb.config.param_scale_regime_1, wandb.config.param_scale_regime_2, wandb.config.param_scale_regime_3]
  wandb.config.init_seeds = np.random.choice(100000000, wandb.config.num_trainings, replace=False)

  updaters = [update_standard, update_mahal, update_vol, update_l2,\
              update_mahal_vol, update_mahal_l2,\
              update_Selective, update_l2_stop, update_l2_stop_selective]

  scaled_glorot_small = partial(variance_scaling, wandb.config.param_scale_regime_2, "fan_avg", "truncated_normal")
  scaled_glorot_large = partial(variance_scaling, wandb.config.param_scale_regime_3, "fan_avg", "truncated_normal")

  # Function to use our architecture
  init_fun_small, predict_fun_small = ResNet50_small(wandb.config.num_classes)
  init_fun_large, predict_fun_large = ResNet50_large(wandb.config.num_classes)
  opt_init, opt_update, get_params = optimizers.sgd(1.0)

  # Loading the dataset
  num_batches, batches, test_batches = datasets.cifar10_loaders(wandb.config.batch_size)

  # Create log file and arrays to track the repeated errors
  logs = log_handler.init_logs(len(wandb.config.training_names), wandb.config.num_epochs, wandb.config.num_eigs_keep)

  print("Set of Updates: ", wandb.config.training_names)

  num_params = 4677872
  track_Q = np.random.rand(num_params, wandb.config.num_eigs_keep)*100
  for train_index in range(len(wandb.config.training_names)):
      for seed in wandb.config.init_seeds:
          print("############################# New "+wandb.config.training_names[train_index]+" Training ###############################")
          rng_key = random.PRNGKey(seed)
          if wandb.config.init_index[train_index] == 1: 
              _, init_params = init_fun_small(rng_key, wandb.config.input_shape)
              predict_fun = predict_fun_small
              step_size = wandb.config.step_size
          elif wandb.config.init_index[train_index] == 2:
              _, init_params = init_fun_large(rng_key, wandb.config.input_shape)
              predict_fun = predict_fun_large
              step_size = wandb.config.step_size
          opt_state = opt_init(init_params)
          updater = updaters[wandb.config.updater_index[train_index]]
          run_logs =  log_handler.init_run_logs() # training, test, time logs, weight norm
          for epoch in range(wandb.config.num_epochs):
              start_time = time.time()
              for _ in range(num_batches):
                opt_state = updater(opt_state, next(batches), epoch)
              epoch_time = time.time() - start_time
              params = get_params(opt_state)
              #norm = np.sum([np.linalg.norm(param_set[0])+np.linalg.norm(param_set[1]) for param_set in params])

              train_acc = accuracy(params, next(batches))
              test_acc = accuracy(params, next(test_batches))
              print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
              print("Training set accuracy {}".format(train_acc))
              print("Test set accuracy {}".format(test_acc))
              wandb.log({"Training Type": wandb.config.training_names[train_index], "epoch": epoch, "epoch_time": epoch_time,\
                         wandb.config.training_names[train_index]+" Train Accuracy": float(train_acc),\
                         wandb.config.training_names[train_index]+" Test Accuracy": float(test_acc)})

              run_logs[0] = np.append(run_logs[0], train_acc)
              run_logs[1] = np.append(run_logs[1], test_acc)
              run_logs[2] = np.append(run_logs[2], epoch_time)
              #run_logs[3] = np.append(run_logs[3], norm)

          logs[0 + (train_index*5)] = np.vstack([logs[0 + (train_index*5)], run_logs[0].reshape(1,wandb.config.num_epochs)])
          logs[1 + (train_index*5)] = np.vstack([logs[1 + (train_index*5)], run_logs[1].reshape(1,wandb.config.num_epochs)])
          logs[2 + (train_index*5)] = np.vstack([logs[2 + (train_index*5)], run_logs[2].reshape(1,wandb.config.num_epochs)])
          converge_eig_vals, track_Q = param_simultaneous_power_iteration(params, wandb.config.num_eigs_keep,\
                  next(batches), track_Q)
          logs[3 + (train_index*5)] = np.vstack([logs[3 + (train_index*5)], converge_eig_vals])
          # logs[4 + (train_index*5)] = np.vstack([logs[4 + (train_index*5)], run_logs[3].reshape(1,wandb.config.num_epochs)])

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
  log_handler.create_all_plots(all_means, all_std, wandb.config.training_names, "Accuracy", "CIFAR-10 Classification", False)
