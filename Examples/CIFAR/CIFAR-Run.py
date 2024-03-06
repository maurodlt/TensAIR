from mpi4py import MPI
import tensair_py
import math
import os
comm = MPI.COMM_WORLD

tensair_path = os.environ.get("TENSAIR_PATH")

mini_batch_size = 128
msg_sec = 150    
train_data_file = tensair_path + "/data/cifar/cifar-10_batch1.bin" #Downloaded from https://www.cs.toronto.edu/~kriz/cifar.html (data_batch_1 from CIFAR-10 binary file)

event_generator = tensair_py.CIFAR_EventGenerator(mpi_comm=comm, tag=1, mini_batch_size=mini_batch_size, 
                                                  msg_sec=msg_sec, train_data_file=train_data_file)

inputMaxSize = 4+4+ (8*2) + (4*mini_batch_size) + (4*mini_batch_size*32*32*3)
gradientsMaxSize = 8+4+4+4+(8*12) + (4*((64*10) + (10) + (3*3*3*32) + (32) + (3*3*32*64) + (64) + (3*3*64*64) + (64) + (1024*64) + (64)))
window_size = max(inputMaxSize, gradientsMaxSize)
saved_model_dir = tensair_path + "/data/cifar/cifar_model.tf"  #Created using CIFAR-Model notebook
drift_detector_mode = 1
dataset_size = 50000
epoch_size= math.floor(dataset_size/mini_batch_size)
model = tensair_py.TensAIR(mpi_comm=comm, tag=2, window_size=window_size, saved_model_dir=saved_model_dir, 
                           drift_detector_mode=drift_detector_mode,epoch_size=epoch_size)

operators = [event_generator, model]
links = [[0,1],[1,1]]
basic_dataflow = tensair_py.BasicDataflow(mpi_comm=comm, operators=operators, links=links)
print("Starting dataflow")
basic_dataflow.streamProcess()