#include "Demo.hpp"
#include "../TensAIR/EventGenerator.hpp"
#include "../TensAIR/TensAIR.hpp"
#include "../TensAIR/DriftDetector.hpp"
#include <tensorflow/c/c_api.h>
#include "../demo/EventGenerator.hpp"
#include <cstdlib>
#include <iostream>

using namespace concept_drift_cifar;

Demo::Demo() : Dataflow() {
    int mini_batch_size = 128;
    int epochs = 1000000;
    int throughput = 200; //messages in a row before waiting 1 sec    
    char* path_value = std::getenv("TENSAIR_PATH");

    //To achieve same results as on python, use full cifar-10 dataset (an aggregation of the 5 databatches from CIFAR-10 binary files)
    std::string train_data_file_str = std::string(path_value) + "/data/cifar/cifar-10_batch1.bin"; //Downloaded from https://www.cs.toronto.edu/~kriz/cifar.html (data_batch_1 from CIFAR-10 binary file)

    const char* train_data_file = train_data_file_str.c_str();
    int drift_frequency = 10;
    generator = new concept_drift_cifar::EventGenerator(1, rank, worldSize, mini_batch_size, throughput, epochs, train_data_file, drift_frequency); //Event Generator operator
    
    std::string saved_model_dir_str = std::string(path_value) + "/data/demo/cifar_model_demo.tf"; //file with tf model created using DEMO-Model notebook
    const char* saved_model_dir = saved_model_dir_str.c_str();
    const char* eval_data_file = "";
    
    size_t inputMaxSize = sizeof(int) + sizeof(int) + (sizeof(size_t)*2) + sizeof(int)*mini_batch_size + sizeof(float)*mini_batch_size*32*32*3; 
    //size of message between TensAIR ranks
    size_t gradientsMaxSize = sizeof(unsigned long int) + sizeof(int) + sizeof(int) + sizeof(int) + sizeof(int) + (sizeof(size_t)*12) + (sizeof(float) * ((64*10) + (10) + (3*3*3*32) + (32) + (3*3*32*64) + (64) + (3*3*64*64) + (64) + (1024*64) + (64)));
    int window_size = (int)max(inputMaxSize,gradientsMaxSize); //max message size involving TensAIR

    int dataset_size = 50000;  //cifar (number of training examples)
    int epoch_size = int(dataset_size/mini_batch_size); //number of mini batches per epoch
    epoch_size = 100;
    TensAIR::Drift_Mode drift_detector_mode=TensAIR::Drift_Mode::AUTOMATIC; //drift detector enabled (using OPTWIN)
    string print_to_folder = "";
    int print_frequency = 1;
    float convergence_factor = 2e-2;
    int epochs_for_convergence = 2;
    int gpus_per_node = 0;
    bool preallocate_tensors = false;

    int broadcast_frequency = 1; //mini batches per broadcast (recommended to set as the world_size)
    model = new TensAIR(2, rank, worldSize, window_size, broadcast_frequency, epochs, gpus_per_node, saved_model_dir, eval_data_file, "serve", epoch_size, convergence_factor, epochs_for_convergence, drift_detector_mode,print_to_folder,print_frequency,preallocate_tensors,mini_batch_size); //TensAIR operator
                
    int drift_window_size = sizeof(int) + sizeof(float);
    int max_widowLoss = 1000;
    std::string file_cuts = std::string(path_value) + "/data/optwin/cut_30-25000_0.01_0.5r.csv"; //file with sliding window splits created using code from https://github.com/maurodlt/OPTWIN
    drift_detector = new drift_detector::DriftDetector(3, rank, worldSize, drift_window_size, max_widowLoss, file_cuts);
    //link operators
    addLink(generator, model);
    addLink(model, model);
    addLink(model, drift_detector);
    addLink(drift_detector, model);

    //init operators
    generator->initialize();
    model->initialize();
    drift_detector->initialize();
    
}

Demo::~Demo() {

	delete generator;
	delete model;
    delete drift_detector;
    
}
