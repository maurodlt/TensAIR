from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import random
import struct
import numpy as np

#Download Cifar dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#images & labels
num_inputs = 2 

#geenrate a drift every drifts_epochs epochs
drifts_epochs = 5 

#currrent epoch
epochs = 0


def shuffle_dataset():
    global train_images, train_labels
    global epochs
    global drifts_epochs
    global global_counter
    
    global_counter = 0
    epochs+=1
    combined = list(zip(train_images, train_labels))
    random.shuffle(combined)
    # Unzip them back into separate lists
    train_images, train_labels = zip(*combined)
    
    #perform concept drift after drifts_epochs epochs
    if epochs % drifts_epochs == 0:
        #select two different labels at random
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        while x == y:
            y = random.randint(0, 9)
            
        # Iterate through each NumPy array in the sequence and swap 2s with 7s and vice versa
        for arr in train_labels:
            # Find indexes of x and y
            idx_x = np.where(arr == x)
            idx_y = np.where(arr == y)

            # Swap x with y and vice versa
            arr[idx_x] = y
            arr[idx_y] = x
    return


shuffle_dataset()

#mini_batch number within current epoch
global_counter = 0

#returns message with format:
'''
    mini_batch_size (4 bytes)
    num_inputs (4 bytes)
    size_inputs_1 (8 bytes)
    ...
    size_inputs_num_inputs ( 8 bytes)
    input_1 (size_inputs_1 byes)
    input_size_inputs_num_inputs (size_inputs_num_inputs bytes) 
'''
def next_message(mini_batch_size):
    global global_counter
    global train_images
    global train_labels
    global num_inputs
    
    message = mini_batch_size.to_bytes(4, 'little') + num_inputs.to_bytes(4, 'little') #add minibatch size to message
    size_inputs = [mini_batch_size*4*32*32*3, mini_batch_size*4] # define size of inputs
    
    #add size of inputs to message
    for inp in size_inputs: 
        message += inp.to_bytes(8, 'little') 
    
    #check for end of training images
    if mini_batch_size*(global_counter+1) > len(train_images)-1:
        shuffle_dataset()
    
    #define minibatch
    minibatch_images = np.array(train_images[mini_batch_size*global_counter:mini_batch_size*(global_counter+1)], dtype=np.float32)
    minibatch_labels = np.array(train_labels[mini_batch_size*global_counter:mini_batch_size*(global_counter+1)], dtype=np.int32)
    
    #serialize images
    message += minibatch_images.tobytes()

    #serialize labels
    message += minibatch_labels.tobytes()
    
    global_counter +=1
    return message