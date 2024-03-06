from mpi4py import MPI
import tensair_py
import os
comm = MPI.COMM_WORLD

tensair_path = os.environ.get("TENSAIR_PATH")

mini_batch_size = 128
msg_sec = 100    
init_code= tensair_path + "/Examples/demo/Demo_Init.py"
event_generator = tensair_py.UDF_EventGenerator(mpi_comm=comm, tag=1, mini_batch_size=mini_batch_size, msg_sec=msg_sec, init_code=init_code)
    
inputMaxSize = 4+4+ (8*2) + (4*mini_batch_size) + (4*mini_batch_size*32*32*3)
gradientsMaxSize = 8+4+4+4+(8*6) + 4 + 4 + (8*((64*10) + (10) + (1024*64) + (64)))
window_size = max(inputMaxSize, gradientsMaxSize)
saved_model_dir = tensair_path + "/data/demo/cifar_model_demo.tf" #Created using DEMO-Model notebook
preallocate_tensors = True
model = tensair_py.TensAIR(mpi_comm=comm, tag=2, window_size=window_size, saved_model_dir=saved_model_dir,preallocate_tensors=preallocate_tensors, mini_batch_size=mini_batch_size)




drift_detector = tensair_py.OPTWIN_drift_detector(mpi_comm=comm, tag=3)

operators = [event_generator, model, drift_detector]
links = [[0,1],[1,1],[1,2],[2,1]]
basic_dataflow = tensair_py.BasicDataflow(mpi_comm=comm, operators=operators, links=links)
print("Starting dataflow")
basic_dataflow.streamProcess()
