import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.api import jit, grad
from jax.scipy.special import logsumexp
from jax.flatten_util import ravel_pytree
import time
import skimage.color as color

def cont_stretch(arr):
	out = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
	return out

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(np.array([x, y]))

def init_random_params(sigma_w, layer_sizes):
  params1 = sigma_w/np.sqrt(layer_sizes[1])*np.random.randn(layer_sizes[0],layer_sizes[1])
  params2 = sigma_w/np.sqrt(layer_sizes[1])*np.random.randn(layer_sizes[1],layer_sizes[2])
  return [(params1, np.random.normal(0.0, sigma_w, (layer_sizes[1]))),\
          (params2, np.random.normal(0.0, sigma_w, (layer_sizes[2])))]

# Computes the sigmoid activation function
def sigmoid(x):
    return jnp.where(x >= 0, 1/(1+jnp.exp(-x)),  jnp.exp(x)/(1+jnp.exp(x)))

# Computes the ReLU activation function
def relu(x):
    return jnp.maximum(x, 0)

def predict(params, inputs):
  outputs = jnp.dot(inputs, params[0][0]) + params[0][1]
  inputs = relu(outputs)
  outputs = jnp.dot(inputs, params[1][0]) + params[1][1]
  return outputs

def accuracy(params, batch):
  inputs, targets = batch
  net_out = jnp.round(predict(params, inputs))
  return jnp.mean(net_out == targets)

# Functions for the different possible pieces of a loss function
def loss(params, batch):
  inputs, targets = batch
  net_out = predict(params, inputs)
  return jnp.sum(jnp.power(net_out - targets, 2))

def loss1(params, batch):
  inputs, targets = batch
  net_out = predict(params, inputs)
  return jnp.sum(jnp.power(net_out - targets, 2))

def mahalanobis_dist(params, batch, params_constant_ravel):
  inputs, targets = batch
  grads = grad(loss)(params, batch)
  grads_flat, _ = ravel_pytree(grads)
  return jnp.dot(params_constant_ravel.T, grads_flat)

def param_volume(params, batch):
  eig_vals, _ = param_simultaneous_power_iteration(params, num_eigs_keep, batch)
  return(-jnp.log(jnp.prod(eig_vals)))

def l2_regularizer(params):
  flat_params, unflatten = ravel_pytree(params)
  return jnp.dot(flat_params.T, flat_params)

# Run experiments
if __name__ == "__main__":
  # Function which updates model parameters without regularization
  @jit
  def update_standard(params, batch, epoch):
    grads = grad(loss)(params, batch)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

  # Function which updates model parameters with mahalanobis dist regularizer
  @jit
  def update_mahal(params, batch, epoch):
    const_flat_params, unflattener = ravel_pytree(params)
    grads = grad(loss)(params, batch)
    mahala_grads = grad(mahalanobis_dist)(params, batch, const_flat_params)
    return [(w-step_size*dw - reg_step*jnp.multiply(jnp.abs(dw_mah),jnp.sign(w)),\
             b-step_size*db - reg_step*jnp.multiply(jnp.abs(db_mah),jnp.sign(b)))\
             for (w, b),(dw, db),(dw_mah, db_mah) in zip(params,grads,mahala_grads)]

  # Function which updates model parameters with full regularizer
  @jit
  def update_vol(params, batch, epoch, init_Q):
    grads = grad(loss)(params, batch)
    volume_grads = grad(param_volume)(params, batch, init_Q)
    return [(w - step_size*dw - vol_step*dw_vol, b - step_size*db - vol_step*db_vol)\
           for (w, b),(dw, db),(dw_vol, db_vol) in zip(params,grads,volume_grads)]

  # Function which updates model parameters with L2 regularization
  @jit
  def update_l2(params, batch, epoch):
    grads = grad(loss)(params, batch)
    reg_grads = grad(l2_regularizer)(params)
    return [(w - step_size*dw - l2_step*dw_reg, b - step_size*db - l2_step*db_reg) \
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
    return [(w - step_size*dw - reg_step*dw_mah - vol_step*dw_vol, \
             b - step_size*db - reg_step*db_mah - vol_step*db_vol) \
           for (w, b),(dw, db),(dw_mah, db_mah),(dw_vol, db_vol) in zip(params,grads,mahala_grads,volume_grads)]

  # Function which updates model parameters with mahalanobis dist regularizer
  @jit
  def update_mahal_l2(params, batch, epoch):
    const_flat_params, unflattener = ravel_pytree(params)
    grads = grad(loss)(params, batch)
    mahala_grads = grad(mahalanobis_dist)(params, batch, const_flat_params)
    reg_grads = grad(l2_regularizer)(params)
    return [(w-step_size*dw - l2_step*dw_reg - reg_step*jnp.multiply(jnp.abs(dw_mah),jnp.sign(w)),\
             b-step_size*db - l2_step*db_reg - reg_step*jnp.multiply(jnp.abs(db_mah),jnp.sign(b)))\
             for (w, b),(dw, db),(dw_mah, db_mah),(dw_reg, db_reg) in zip(params,grads,mahala_grads,reg_grads)]

  @jit
  def update_Selective(params, batch, epoch):
    const_flat_params, unflattener = ravel_pytree(params)
    important_params = grad(mahalanobis_dist)(params, batch, const_flat_params)
    grads = grad(loss)(params, batch)
    reg_grads = grad(l2_regularizer)(params)                                                            #0.01
    return [(w-step_size*dw-selective_l2_step*jnp.where(jnp.abs(dw_choose)>0.05*jnp.max(dw_choose),0.0,dw_reg),\
             b-step_size*db-selective_l2_step*jnp.where(jnp.abs(db_choose)>0.05*jnp.max(db_choose),0.0,db_reg))\
             for (w, b),(dw, db),(dw_reg, db_reg),(dw_choose, db_choose) in zip(params,grads,reg_grads,important_params)]

  # Function which updates model parameters with L2 regularization
  #@jit
  def update_l2_stop(params, batch, epoch):
    if epoch < int(0.7*num_epochs):
        return update_l2(params, batch, epoch)
    else:
        return update_standard(params, batch, epoch)

  #@jit
  def update_l2_stop_selective(params, batch, epoch):
    if epoch < int(0.7*num_epochs):
        return update_l2(params, batch, epoch)
    else:
        return update_Selective(params, batch, epoch)

  # Set hyper-parameters
  layer_sizes = [2, 12, 1] 
  sigma_w = 0.1 
  step_size = 1.5*0.0008/sigma_w/0.1
  reg_step = 0.0001 
  l2_step = 0.001
  selective_l2_step = 0.1
  num_epochs = 400  
  training_names = ['SGD','Metric','L2','Selective_L2','L2_Stopped','Selective_L2_Stopped']
  updater_index = [0, 1, 3, 6, 7, 8]
  updaters = [update_standard, update_mahal, update_vol, update_l2,\
              update_mahal_vol, update_mahal_l2,\
              update_Selective, update_l2_stop, update_l2_stop_selective]

  train_images = np.array([[-1,-1],[-1,1],[1,-1],[1,1]]).astype(np.float64)
  train_labels = np.array([-1,1,1,-1]).astype(np.float64).reshape(4,1)
  batch = (train_images, train_labels)

  init_params = init_random_params(sigma_w, layer_sizes)
  
  for train_index in range(len(training_names)):
    print("############################# New "+training_names[train_index]+" Training ###############################")
    params = init_params.copy()
    updater = updaters[updater_index[train_index]]
    for epoch in range(num_epochs):
	    start_time = time.time()
	    params = updater(params, batch, epoch)
	    epoch_time = time.time() - start_time

	    train_acc = accuracy(params, (train_images, train_labels))
	    train_loss = loss1(params, batch)
	    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
	    print("Training set accuracy {}".format(train_acc))
	    print("Loss: {}".format(train_loss))
	    if epoch % 1 == 0:
	      origin = [0], [0]
	      plot_vecs = params[0][0].T
	      q_color = np.hstack([cont_stretch(params[1][0]), np.zeros(params[1][0].shape), np.zeros(params[1][0].shape)])
	      q = plt.quiver(*origin, plot_vecs[:,0], plot_vecs[:,1], scale=10, color=q_color, cmap='cividis')
	      q.set_array(np.linspace(np.min(params[1][0]),np.max(params[1][0]),10))
	      plt.colorbar(q, cmap='cividis')
	      plt.scatter(train_images[:,0], train_images[:,1], color=['red','blue','green','orange'])
	      plt.title("Train Accuracy: "+str(train_acc)+"; Train Loss: "+str(train_loss))
	      plt.savefig(training_names[train_index]+'/'+str(epoch)+'.png')
	      plt.close()
