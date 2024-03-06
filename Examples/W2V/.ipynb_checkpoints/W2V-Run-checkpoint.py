from mpi4py import MPI
import tensair_py
import math
import os
comm = MPI.COMM_WORLD

tensair_path = os.environ.get("TENSAIR_PATH")

mini_batch_size = 2048
msg_sec = 200    
train_data_file = tensair_path + "/data/w2v/shakespeare_train.txt" #Created using W2V_data notebook
event_generator = tensair_py.W2V_EventGenerator(mpi_comm=comm, tag=1, mini_batch_size=mini_batch_size, msg_sec=msg_sec, train_data_file=train_data_file)

inputMaxSize = 4+4+ (8*3) + (4*mini_batch_size) + (4*mini_batch_size*5) + (4*mini_batch_size*5)
gradientsMaxSize = 8+4+4+4+(8*4) + 4 + 4 + (4*50000*300*2)
window_size = max(inputMaxSize, gradientsMaxSize)
saved_model_dir = tensair_path + "/data/w2v/w2v_model.tf"  #Created using W2V-Model notebook
drift_detector_mode = 1
dataset_size = 112640 #shakespeare dataset size
epoch_size= math.floor(dataset_size/mini_batch_size)
model = tensair_py.TensAIR(mpi_comm=comm, tag=2, window_size=window_size, saved_model_dir=saved_model_dir, 
                           drift_detector_mode=drift_detector_mode,epoch_size=epoch_size)

operators = [event_generator, model]
links = [[0,1],[1,1]]
basic_dataflow = tensair_py.BasicDataflow(mpi_comm=comm, operators=operators, links=links)
print("Starting dataflow")
basic_dataflow.streamProcess()
