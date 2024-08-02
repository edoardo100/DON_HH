import sys
import os
import numpy as np
import matplotlib.pyplot as plt

fontsize_ticks = 20
fontsize_title = 25
fontsize_axis  = 22

colors = ['#f19066',
          '#faded1',
          '#546de5',
          '#d3d9f8']

def read_losses(file_path):
    # Read the data from the file
    epochs = []
    train_losses = []
    test_losses = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Epoch:'):
                parts = line.split()
                epoch = int(parts[1])
                train_loss = float(parts[5])
                test_loss = float(parts[7])
                epochs.append(epoch)
                train_losses.append(train_loss)
                test_losses.append(test_loss)

    return np.array(epochs), np.array(train_losses), np.array(test_losses) 

if __name__ == "__main__":
    # Check if a file path argument is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    # Get the file path from the command line argument
    file_path = sys.argv[1]

    base_name, extension = os.path.splitext(file_path)

    # Generate the list of file paths
    file_paths = [file_path, f"{base_name}_2{extension}", f"{base_name}_3{extension}"]

    # Read the data from the three files
    epochs_list = []
    train_losses_list = []
    test_losses_list = []

    for file_path in file_paths:
        epochs, train_losses, test_losses = read_losses(file_path)
        epochs_list.append(epochs)
        train_losses_list.append(train_losses)
        test_losses_list.append(test_losses)

    # Convert lists to numpy arrays
    train_losses_array = np.vstack(train_losses_list)
    test_losses_array = np.vstack(test_losses_list)

    # Calculate the mean and standard deviation
    train_losses_mean = np.mean(train_losses_array, axis=0)
    train_losses_std  = np.std(train_losses_array, axis=0)
    test_losses_mean  = np.mean(test_losses_array, axis=0)
    test_losses_std   = np.std(test_losses_array, axis=0)

    # Create a semilogy plot
    plt.figure(figsize=(8.5, 6.5))
    plt.semilogy(epochs, train_losses_list[0], label='Train Loss',linewidth=2.0, c=colors[0])
    plt.semilogy(epochs, test_losses_list[0] , label='Test Loss' ,linewidth=2.0, c=colors[2])
    plt.semilogy(epochs, train_losses_mean, '--', label='Train Loss mean',linewidth=1.0, c=colors[0])
    plt.semilogy(epochs, test_losses_mean , '--', label='Test Loss mean' ,linewidth=1.0, c=colors[2])
    plt.fill_between(epochs, train_losses_mean-train_losses_std, train_losses_mean+train_losses_std,
    alpha=0.5, edgecolor=colors[1], facecolor=colors[1])
    plt.fill_between(epochs, test_losses_mean-test_losses_std, test_losses_mean+test_losses_std,
    alpha=0.5, edgecolor=colors[3], facecolor=colors[3])  
    plt.xlabel('Epoch',fontsize=fontsize_axis)
    plt.ylabel('Loss' ,fontsize=fontsize_axis)
    arch = ''
    if 'don' in file_path:
        arch = 'don'
        plt.title('DON',fontsize=fontsize_title)
    elif 'fno' in file_path:
        arch = 'fno'
        plt.title('FNO',fontsize=fontsize_title)
        plt.xticks([])
        plt.xticks([0,500,1000,1500,2000])
    elif 'wno' in file_path:
        arch = 'wno'
        plt.title('WNO',fontsize=fontsize_title)
        plt.xticks([])
        plt.xticks([0,500,1000,1500,2000])
    else:
        plt.title('Relative L2 error')
    plt.legend(fontsize=fontsize_axis)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(True)
    plt.savefig(arch+"_loss.eps",format='eps')
    plt.show()