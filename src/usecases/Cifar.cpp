#include "Cifar.hpp"
#include "../cifar/EventGenerator.hpp"
#include "../TensAIR/TensAIR.hpp"
#include "../TensAIR/DriftDetector.hpp"
#include <tensorflow/c/c_api.h>

using namespace cifar;

Cifar::Cifar() : Dataflow() {
    int mini_batch_size = 256;
    int epochs = 300;
    int broadcast_frequency = 1; //mini batches per message (IN DEVELOPMENT, USE 1)
    int gpus_per_node = 0;
    int throughput = 100; //messages in a row before waiting 1 sec
    const char* train_data_file = "../data/CIFAR/cifar-train.txt"; //file with trining data

    generator = new cifar::EventGenerator(1, rank, worldSize, mini_batch_size, throughput, epochs, train_data_file); //Event Generator operator
    
    const char* saved_model_dir = "../data/CIFAR/python_interface/cifar.tf"; //file with tf model
    const char* eval_data_file = "../data/CIFAR/cifar-evaluate.bytes"; //file with evaluation data (parsed to binary file)
    //size of message between EventGenerator and TensAIR
    size_t inputMaxSize = sizeof(int) + sizeof(int) + (sizeof(size_t)*2) + sizeof(int)*mini_batch_size + sizeof(float)*mini_batch_size*32*32*3; 
    //size of message between TensAIR ranks
    size_t gradientsMaxSize = sizeof(unsigned long int) + sizeof(int) + sizeof(int) + sizeof(int) + (sizeof(size_t)*12) + (sizeof(float) * ((64*10) + (10) + (3*3*3*32) + (32) + (3*3*32*64) + (64) + (3*3*64*64) + (64) + (1024*64) + (64)));
    int window_size = (int)max(inputMaxSize,gradientsMaxSize); //max message size involving TensAIR
    int dataset_size = 50000;  //cifar (number of training examples)
    int epoch_size = int(dataset_size/mini_batch_size); //number of mini batches per epoch
    TensAIR::Drift_Mode drift_detector_mode=TensAIR::Drift_Mode::AUTOMATIC; //drift detector enabled
    string print_to_folder = "../output/";
    int print_frequency = 1;
    float convergence_factor = 1e-4;
    int epochs_for_convergence = 5;
    model = new TensAIR(2, rank, worldSize, window_size, broadcast_frequency, epochs, gpus_per_node, saved_model_dir, eval_data_file, "serve", epoch_size, convergence_factor, epochs_for_convergence, drift_detector_mode,print_to_folder,print_frequency); //TensAIR operator
    drift_detector = new drift_detector::DriftDetector(3, rank, worldSize);

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

Cifar::~Cifar() {

	delete generator;
	delete model;
    
}
