#include "Cifar.hpp"
#include "../cifar/EventGenerator.hpp"
#include "../TensAIR/TensAIR.hpp"
#include "../TensAIR/DriftDetector.hpp"
#include <tensorflow/c/c_api.h>
#include <cstdlib>
#include <iostream>

using namespace cifar;

Cifar::Cifar() : Dataflow() {
    int mini_batch_size = 128;
    int epochs = 300;
    int throughput = 150; //messages in a row before waiting 1 sec
    char* path_value = std::getenv("TENSAIR_PATH");
    std::string train_data_file_str = std::string(path_value) + "/data/cifar/cifar-10_batch1.bin"; //Downloaded from https://www.cs.toronto.edu/~kriz/cifar.html (data_batch_1 from CIFAR-10 binary file)
    const char* train_data_file = train_data_file_str.c_str();

    generator = new cifar::EventGenerator(1, rank, worldSize, mini_batch_size, throughput, epochs, train_data_file); //Event Generator operator
    
    int gpus_per_node = 0;

    int broadcast_frequency = 1; //mini batches per broadcast (recommended to set as the world_size)
    std::string saved_model_dir_str = std::string(path_value) + "/data/cifar/cifar_model.tf"; //file with tf model created using CIFAR-Model notebook
    const char* saved_model_dir = saved_model_dir_str.c_str();
    const char* eval_data_file = ""; //file with evaluation data (parsed to binary file)
    //size of message between EventGenerator and TensAIR
    size_t inputMaxSize = sizeof(int) + sizeof(int) + (sizeof(size_t)*2) + sizeof(int)*mini_batch_size + sizeof(float)*mini_batch_size*32*32*3; 
    //size of message between TensAIR ranks
    size_t gradientsMaxSize = sizeof(unsigned long int) + sizeof(int) + sizeof(int) + sizeof(int) + (sizeof(size_t)*12) + (sizeof(float) * ((64*10) + (10) + (3*3*3*32) + (32) + (3*3*32*64) + (64) + (3*3*64*64) + (64) + (1024*64) + (64)));
    int window_size = (int)max(inputMaxSize,gradientsMaxSize); //max message size involving TensAIR
    int dataset_size = 50000;  //cifar (number of training examples)
    int epoch_size = int(dataset_size/mini_batch_size); //number of mini batches per epoch
    TensAIR::Drift_Mode drift_detector_mode=TensAIR::Drift_Mode::ALWAYS_TRAIN; //drift detector disabled 
    int print_frequency = 1;
    float convergence_factor = 1e-4;
    int epochs_for_convergence = 5;
    std::string print_to_folder = "";
    model = new TensAIR(2, rank, worldSize, window_size, broadcast_frequency, epochs, gpus_per_node, saved_model_dir, eval_data_file, "serve", epoch_size, convergence_factor, epochs_for_convergence, drift_detector_mode,print_to_folder,print_frequency); //TensAIR operator

    //link operators
    addLink(generator, model);
    addLink(model, model);

    //init operators
    generator->initialize();
    model->initialize();
}

Cifar::~Cifar() {

	delete generator;
	delete model;
    
}
