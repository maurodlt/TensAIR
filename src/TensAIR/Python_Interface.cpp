#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <mpi.h>
#include <mpi4py/mpi4py.h>


#include "../word_embedding/EventGenerator.hpp"
#include "../cifar/EventGenerator.hpp"
#include "TensAIR.hpp"
#include "EventGenerator.hpp"
#include "../dataflow/BasicDataflow.hpp"
#include "DriftDetector.hpp"

#include "../demo/EventGenerator.hpp"


namespace py = pybind11;

//Parse py_comm to MPI_Comm
MPI_Comm *get_mpi_comm(py::object py_comm) {
  auto comm_ptr = PyMPIComm_Get(py_comm.ptr());

  if (!comm_ptr)
    throw py::error_already_set();
    
  return comm_ptr;
}

// Set PyBind module
PYBIND11_MODULE(tensair_py, m) {
    if (import_mpi4py() < 0) throw py::error_already_set();

    py::class_<Vertex>(m, "Vertex")
        .def(py::init([](py::object py_comm, const int tag, int window_size) {
            MPI_Comm* comm = get_mpi_comm(py_comm);
            int world_size; MPI_Comm_size(*comm, &world_size); //get world_size from MPI_Comm
            int rank; MPI_Comm_rank(*comm, &rank); // get rank from MPI_Comm

            return std::unique_ptr<Vertex>(new Vertex(tag, rank, world_size, window_size, *comm));
        }), py::arg("mpi_comm"), py::arg("tag"), py::arg("window_size")=1000000);

    py::class_<BasicVertex<>, Vertex>(m, "BasicVertex")
        .def(py::init([](py::object py_comm, const int tag, int window_size) {
            MPI_Comm* comm = get_mpi_comm(py_comm);
            int world_size; MPI_Comm_size(*comm, &world_size); //get world_size from MPI_Comm
            int rank; MPI_Comm_rank(*comm, &rank); // get rank from MPI_Comm

            return std::unique_ptr<BasicVertex<>>(new BasicVertex<>(tag, rank, world_size, window_size, *comm));
        }), py::arg("mpi_comm"), py::arg("tag"), py::arg("window_size")=1000000);
    
    
    py::class_<word_embedding::EventGenerator, BasicVertex<>, Vertex>(m, "W2V_EventGenerator")
        .def(py::init([](py::object py_comm, const int tag, int mini_batch_size, int msg_sec, int epochs, const char* train_data_file, int window_size) {
            MPI_Comm* comm = get_mpi_comm(py_comm);
            int world_size; MPI_Comm_size(*comm, &world_size); //get world_size from MPI_Comm
            int rank; MPI_Comm_rank(*comm, &rank); // get rank from MPI_Comm

            return std::unique_ptr<word_embedding::EventGenerator>(new word_embedding::EventGenerator(tag, rank, world_size, mini_batch_size, msg_sec, epochs, train_data_file, window_size, *comm));
        }), py::arg("mpi_comm"), py::arg("tag"), py::arg("mini_batch_size"), py::arg("msg_sec") = 1000, py::arg("epochs")=INT_MAX, py::arg("train_data_file"), py::arg("window_size")=1000000);
    
    
    py::class_<cifar::EventGenerator, BasicVertex<>, Vertex>(m, "CIFAR_EventGenerator")
        .def(py::init([](py::object py_comm, const int tag, int mini_batch_size, int msg_sec, int epochs, const char* train_data_file, int window_size) {
            MPI_Comm* comm = get_mpi_comm(py_comm);
            int world_size; MPI_Comm_size(*comm, &world_size); //get world_size from MPI_Comm
            int rank; MPI_Comm_rank(*comm, &rank); // get rank from MPI_Comm

            return std::unique_ptr<cifar::EventGenerator>(new cifar::EventGenerator(tag, rank, world_size, mini_batch_size, msg_sec, epochs, train_data_file, window_size, *comm));
        }), py::arg("mpi_comm"), py::arg("tag"), py::arg("mini_batch_size"), py::arg("msg_sec") = 1000, py::arg("epochs")=INT_MAX, py::arg("train_data_file"), py::arg("window_size")=1000000);


    py::class_<event_generator::EventGenerator, BasicVertex<>, Vertex>(m, "UDF_EventGenerator")
        .def(py::init([](py::object py_comm, const int tag, int mini_batch_size, int msg_sec, int window_size, std::string init_code) {
            MPI_Comm* comm = get_mpi_comm(py_comm);
            int world_size; MPI_Comm_size(*comm, &world_size); //get world_size from MPI_Comm
            int rank; MPI_Comm_rank(*comm, &rank); // get rank from MPI_Comm

            return std::unique_ptr<event_generator::EventGenerator>(new event_generator::EventGenerator(tag, rank, world_size, mini_batch_size, msg_sec, window_size, init_code, *comm));
        }), py::arg("mpi_comm"), py::arg("tag"), py::arg("mini_batch_size"), py::arg("msg_sec") = 100, py::arg("window_size")=1000000, py::arg("init_code")="");

    py::class_<concept_drift_cifar::EventGenerator, BasicVertex<>, Vertex>(m, "Fixed_EventGenerator")
        .def(py::init([](py::object py_comm, const int tag, int mini_batch_size, int msg_sec, int epochs, const char* train_data_file, int window_size, int drift_frequency) {
            MPI_Comm* comm = get_mpi_comm(py_comm);
            int world_size; MPI_Comm_size(*comm, &world_size); //get world_size from MPI_Comm
            int rank; MPI_Comm_rank(*comm, &rank); // get rank from MPI_Comm

            return std::unique_ptr<concept_drift_cifar::EventGenerator>(new concept_drift_cifar::EventGenerator(tag, rank, world_size, mini_batch_size, msg_sec, epochs, train_data_file, window_size, drift_frequency, *comm));
        }), py::arg("mpi_comm"), py::arg("tag"), py::arg("mini_batch_size"), py::arg("msg_sec") = 1000, py::arg("epochs")=INT_MAX, py::arg("train_data_file"), py::arg("window_size")=1000000, py::arg("drift_frequency")=10);

    
    py::class_<drift_detector::DriftDetector, BasicVertex<>, Vertex>(m, "OPTWIN_drift_detector")
    .def(py::init([](py::object py_comm, const int tag, int windowSize, int max_widowLoss, const char* file_cuts) {
        MPI_Comm* comm = get_mpi_comm(py_comm);
        int world_size; MPI_Comm_size(*comm, &world_size); //get world_size from MPI_Comm
        int rank; MPI_Comm_rank(*comm, &rank); // get rank from MPI_Comm
        std::string file_cuts_str(file_cuts);

        return std::unique_ptr<drift_detector::DriftDetector>(new drift_detector::DriftDetector(tag, rank, world_size, windowSize, max_widowLoss, file_cuts_str, *comm));
    }), py::arg("mpi_comm"), py::arg("tag"), py::arg("windowSize") = sizeof(int) + sizeof(float), py::arg("max_widowLoss") = 1000, py::arg("file_cuts")="");
    
    
    py::class_<TensAIR, BasicVertex<>, Vertex>(m, "TensAIR")
        .def(py::init([](py::object py_comm, const int tag, int window_size, int broadcast_frequency, int epochs, int gpus_per_node, const char* saved_model_dir, const char* eval_data_file, const char* tags, int epoch_size, float convergence_factor, int epochs_for_convergence, int drift_detector_mode, string print_to_folder, int print_frequency, bool preallocate_tensors, int mini_batch_size) {
            MPI_Comm* comm = get_mpi_comm(py_comm);
            int world_size; MPI_Comm_size(*comm, &world_size); //get world_size from MPI_Comm
            int rank; MPI_Comm_rank(*comm, &rank); // get rank from MPI_Comm            

            TensAIR::Drift_Mode drift_mode;

            if(drift_detector_mode == 0){
                drift_mode = TensAIR::Drift_Mode::AUTOMATIC;
            }else if(drift_detector_mode == 1){
                drift_mode = TensAIR::Drift_Mode::ALWAYS_TRAIN;
            }else{
                drift_mode = TensAIR::Drift_Mode::NEVER_TRAIN;
            }
            
           
            return std::unique_ptr<TensAIR>(new TensAIR(tag, rank, world_size, window_size, broadcast_frequency, epochs, gpus_per_node, saved_model_dir, eval_data_file, tags, epoch_size, convergence_factor, epochs_for_convergence, drift_mode, print_to_folder, print_frequency, preallocate_tensors, mini_batch_size, *comm));
        }), py::arg("mpi_comm"), py::arg("tag"), py::arg("window_size"), py::arg("broadcast_frequency")=1, py::arg("epochs")=INT_MAX, py::arg("gpus_per_node") = 0, py::arg("saved_model_dir"), py::arg("eval_data_file")="", py::arg("tags") = "serve", py::arg("epoch_size") = 100, py::arg("convergence_factor") = 2e-2, py::arg("epochs_for_convergence") = 2, py::arg("drift_detector_mode") = 0, py::arg("print_to_folder")="", py::arg("print_frequency") = 1, py::arg("preallocate_tensors") = false, py::arg("mini_batch_size") = 0)
        .def_readwrite("predict_input_dims", &TensAIR::predict_input_dims)
        .def_readwrite("predict_output_dims", &TensAIR::predict_output_dims)
        .def_readwrite("retrieve_delta_output_dims", &TensAIR::retrieve_delta_output_dims)
        .def_readwrite("evaluate_input_dims", &TensAIR::evaluate_input_dims)
        .def_readwrite("evaluate_output_dims", &TensAIR::evaluate_output_dims);

    py::class_<Dataflow>(m, "Dataflow")
        .def(py::init([](py::object py_comm) {
            MPI_Comm* comm = get_mpi_comm(py_comm);
            return std::unique_ptr<Dataflow>(new Dataflow(*comm));
        }), py::arg("mpi_comm"));
    
    py::class_<BasicDataflow, Dataflow>(m, "BasicDataflow")
        .def(py::init([](py::object py_comm, std::vector<py::object> &operators, std::vector<std::tuple<int,int>> &operatorsLinks) {
            MPI_Comm* comm = get_mpi_comm(py_comm);

            //Parse py::object to Vertex*
            std::vector<Vertex*> ops;
            for(int i = 0; i < operators.size(); i++){
                Vertex* op = operators[i].cast<Vertex*>();
                ops.push_back(op);
            }

            return std::unique_ptr<BasicDataflow>(new BasicDataflow(*comm, ops, operatorsLinks));
        }), py::arg("mpi_comm"), py::arg("operators"), py::arg("links"))
        .def("streamProcess", &BasicDataflow::streamProcess);
    
}

