from mpi4py import MPI
import os
import sys
sys.path.append('/home/su_dalle-lucca-tosi/GitHub/TensAIR/lib')
import tensair_py

comm = MPI.COMM_WORLD

tensair_path = os.environ.get("TENSAIR_PATH")

mini_batch_size = 32
msg_sec = 100    
train_data_file = "/home/su_dalle-lucca-tosi/TensAIR_experiments/test_data.bin"
event_generator = tensair_py.AIRPLANE_EventGenerator(mpi_comm=comm, tag=1, mini_batch_size=mini_batch_size, 
                                                  msg_sec=msg_sec, train_data_file=train_data_file)

inputMaxSize = 4+4+ (5*4) + (5*4*mini_batch_size)
gradientsMaxSize = 111108*4
window_size = max(inputMaxSize, gradientsMaxSize)
saved_model_dir = "/home/su_dalle-lucca-tosi/TensAIR_experiments/model"
preallocate_tensors = True
output_folder = "/home/su_dalle-lucca-tosi/TensAIR_experiments/output/"
model = tensair_py.TensAIR(mpi_comm=comm, tag=2, window_size=window_size, saved_model_dir=saved_model_dir,preallocate_tensors=preallocate_tensors, mini_batch_size=mini_batch_size,print_to_folder=output_folder, epoch_size=64, convergence_factor=0.3)


drift_detector = tensair_py.OPTWIN_drift_detector(mpi_comm=comm, tag=3)

operators = [event_generator, model, drift_detector]
links = [[0,1],[1,1],[1,2],[2,1]]
basic_dataflow = tensair_py.BasicDataflow(mpi_comm=comm, operators=operators, links=links)
print("Starting dataflow")
basic_dataflow.streamProcess()
