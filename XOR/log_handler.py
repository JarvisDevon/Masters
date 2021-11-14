import numpy as np
import matplotlib.pyplot as plt

def init_logs(num_log_sets, num_epochs, num_eigs):
  # Create log file and arrays to track the repeated errors
  logs = []
  for i in range(num_log_sets):
      logs.append(np.zeros((1,num_epochs))) # training erros
      logs.append(np.zeros((1,num_epochs))) # test errors
      logs.append(np.zeros((1,num_epochs))) # times
      logs.append(np.zeros((1,num_eigs))) # Eigen-values
      logs.append(np.zeros((1,num_epochs))) # norms
  return logs

def init_run_logs():
  return [np.array([]), np.array([]), np.array([]), np.array([])]

def remove_place_holders(logs):
  # Remove place holder first line of zeros
  new_logs = []
  for log in logs:
      new_logs.append(log[1:])
  return new_logs
        
def calc_all_means(logs, train_names):
  # Calculate the means of the train and test errors for all types of training
  i = 0
  log_sets = []
  for name in train_names:
      new_set = []
      new_set.append(np.mean(logs[i], axis=0))
      new_set.append(np.mean(logs[i+1], axis=0))
      new_set.append(np.mean(logs[i+2]))
      new_set.append(np.mean(logs[i+3], axis=0))
      new_set.append(np.mean(logs[i+4], axis=0))
      i = i + 5

      # Print to the log
      print(name+"_error_train_mean", new_set[0][-1])
      print(name+"_error_test_mean", new_set[1][-1])
      print(name+"_time_mean", new_set[2])
      log_sets.append(new_set)
  return log_sets

def calc_all_std(logs, train_names):
  # Calculate standard deviation of the train and test errors for both types of training
  i = 0
  log_sets = []
  for name in train_names:
      new_set = []
      new_set.append(np.std(logs[i], axis=0))
      new_set.append(np.std(logs[i+1], axis=0))
      new_set.append(np.std(logs[i+2]))
      new_set.append(np.std(logs[i+3], axis=0))
      new_set.append(np.std(logs[i+4], axis=0))
      i = i + 5

      # Print to the log
      print(name+"_error_train_std", new_set[0][-1])
      print(name+"_error_test_std", new_set[1][-1])
      print(name+"_time_std", new_set[2])
      log_sets.append(new_set)
  return log_sets

def create_all_plots(mean_sets, std_sets, set_names, y_label, train_name, regression):

  #############################################################################################################################
  # Plotting Time Graph
  plt.bar(range(len(set_names)), [set[2] for set in mean_sets], width=0.5, color='r', yerr=[set[2] for set in std_sets])
  plt.xticks(range(len(set_names)), set_names)
  plt.title("Mean Time for Each Update Type")
  plt.ylabel("Time (seconds)")
  plt.xlabel("Training Type")
  plt.savefig("time.svg")
  plt.close()

  # Plotting Errors
  start_ep = 1
  full_range = mean_sets[0][0].shape[0]
  part_range = full_range - 200
  if regression == True:
    inset_params = [0.35, 0.5, 0.4, 0.4] #[0.55,0.5,0.4,0.4]
    y_low_lim = 0.0
    y_upper_lim = 30.0
    weight_low_lim = 0
    weight_upper_lim = 50
  else:
    inset_params = [0.55, 0.1, 0.4, 0.4]
    y_low_lim = 0.8
    y_upper_lim = 1.0
    weight_low_lim = 0
    weight_upper_lim = 300
  inset_params_norm = [0.55, 0.3, 0.4, 0.4]
  
  # Plot train errors for both standard and regularized types of training
  train_file = open("train_file.txt", "w")
  fig, ax = plt.subplots(figsize=(12,6))
  axins = ax.inset_axes(inset_params)
  for i in range(len(mean_sets)):
      train_file.write(str(mean_sets[i][0])+"\n")
      train_file.write(str(std_sets[i][0])+"\n")
      markers,caps,bars=ax.errorbar(np.arange(start_ep,full_range,1),mean_sets[i][0][start_ep:],2.0*std_sets[i][0][start_ep:],label=set_names[i])
      [bar.set_alpha(0.5) for bar in bars]
      markers, caps, bars = axins.errorbar(np.arange(part_range,full_range,1), mean_sets[i][0][part_range:],2.0*std_sets[i][0][part_range:])
      [bar.set_alpha(0.5) for bar in bars]
  ax.indicate_inset_zoom(axins, linewidth=3)
  ax.set_title(train_name+": Train "+y_label)
  ax.set_ylabel(y_label)
  axins.set_ylim([y_low_lim-5.0,y_upper_lim])
  ax.set_xlabel("Epochs")
  ax.legend(loc=1, bbox_to_anchor=(1.5,0.6)) #(1.13, 0.6)
  plt.savefig("train_errors.svg")
  plt.close()
  train_file.close()

  # Plot test errors for both standard and regularized types of testing
  test_file = open("test_file.txt", "w")
  fig, ax = plt.subplots(figsize=(12,6))
  axins = ax.inset_axes(inset_params)
  for i in range(len(mean_sets)):
      test_file.write(str(mean_sets[i][1])+"\n")
      test_file.write(str(std_sets[i][1])+"\n")
      markers,caps,bars=ax.errorbar(np.arange(start_ep,full_range,1),mean_sets[i][1][start_ep:],2.0*std_sets[i][1][start_ep:], label=set_names[i])
      [bar.set_alpha(0.5) for bar in bars]
      markers, caps, bars = axins.errorbar(np.arange(part_range,full_range,1), mean_sets[i][1][part_range:],2.0*std_sets[i][1][part_range:])
      [bar.set_alpha(0.5) for bar in bars]
  ax.indicate_inset_zoom(axins, linewidth=3)
  ax.set_title(train_name+": Test "+y_label)
  ax.set_ylabel(y_label)
  axins.set_ylim([y_low_lim,y_upper_lim])
  ax.set_xlabel("Epochs")
  ax.legend(loc=1, bbox_to_anchor=(1.13,0.6))
  plt.savefig("test_error.svg")
  plt.close()
  test_file.close()

  # Plot test errors for both standard and regularized types of testing
  fig, ax = plt.subplots(figsize=(12,6))
  axins = ax.inset_axes(inset_params)
  for i in range(len(mean_sets)):
      markers, caps, bars = ax.errorbar(np.arange(start_ep,full_range,1), 0.5*(mean_sets[i][0][start_ep:]+mean_sets[i][1][start_ep:]),\
                                                         std_sets[i][0][start_ep:]+std_sets[i][1][start_ep:], label=set_names[i])
      [bar.set_alpha(0.5) for bar in bars]
      markers, caps, bars=axins.errorbar(np.arange(part_range,full_range,1), 0.5*(mean_sets[i][0][part_range:]+mean_sets[i][1][part_range:]),\
                                                         std_sets[i][0][part_range:]+std_sets[i][1][part_range:])
      [bar.set_alpha(0.5) for bar in bars]

  ax.indicate_inset_zoom(axins, linewidth=3)
  ax.set_title(train_name+": Average "+y_label)
  ax.set_ylabel(y_label)
  axins.set_ylim([y_low_lim,y_upper_lim])
  ax.set_xlabel("Epochs")
  ax.legend(loc=1, bbox_to_anchor=(1.13,0.6))
  plt.savefig("average_error.svg")
  plt.close()

  # Plot eig value for standard type of testing
  eigs_file = open("eigs_file.txt", "w")
  for i in range(len(mean_sets)):
      fig, ax = plt.subplots(figsize=(12,6))
      eigs_file.write(str(mean_sets[i][3])+"\n")
      eigs_file.write(str(std_sets[i][3])+"\n")
      plt.bar(np.arange(mean_sets[i][3].shape[0]), mean_sets[i][3], width=0.5, color='r', yerr=2.0*std_sets[i][3])
      ax.set_title(train_name+" Eigenvalues: "+set_names[i])
      ax.set_ylabel("Eigenvalue")
      ax.set_xlabel("Order of Eigenvalue")
      ax.legend()
      plt.savefig("eigs_"+set_names[i]+".svg")
      plt.close()
  eigs_file.close()

  # Plot norms of the weights for different training types
  norms_file = open("norms_file.txt", "w")
  fig, ax = plt.subplots(figsize=(12,6))
  axins = ax.inset_axes(inset_params_norm)
  for i in range(len(mean_sets)):
      norms_file.write(str(mean_sets[i][4])+"\n")
      norms_file.write(str(std_sets[i][4])+"\n")
      markers, caps, bars = ax.errorbar(np.arange(start_ep,full_range,1), mean_sets[i][4][start_ep:], 2.0*std_sets[i][4][start_ep:], label=set_names[i])
      [bar.set_alpha(0.5) for bar in bars]
      markers, caps, bars = axins.errorbar(np.arange(part_range,full_range,1), mean_sets[i][4][part_range:],2.0*std_sets[i][4][part_range:])
      [bar.set_alpha(0.5) for bar in bars]
  ax.indicate_inset_zoom(axins, linewidth=3)
  ax.set_title(train_name+": Norm of Weights")
  axins.set_ylim([weight_low_lim,weight_upper_lim])
  ax.set_ylabel("Weight norm")
  ax.set_xlabel("Epochs")
  ax.legend(loc=1, bbox_to_anchor=(1.13,0.6))
  plt.savefig("norms.svg")
  plt.close()
  norms_file.close()

