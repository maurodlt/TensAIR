#include "VGG16.hpp"
#include "../vgg16_Convergence/VGG16_Convergence.hpp"
#include "../TensAIR/TensAIR.hpp"
#include <tensorflow/c/c_api.h>
#include <cstdlib>
#include <iostream>

using namespace vgg16;

VGG16::VGG16(int mini_batch_size, int epochs, int gpus_per_node, float loss_objective) : Dataflow() {
    char* path_value = std::getenv("TENSAIR_PATH");
    int broadcast_frequency = worldSize; //mini batches per broadcast (recommended to set as the world_size)
    std::string saved_model_dir_str = std::string(path_value) + "/data/vgg16/vgg16_model.tf";
    const char* saved_model_dir = saved_model_dir_str.c_str();
    const char* eval_data_file = ""; //file with evaluation data (parsed to binary file)
    
    //size of message between TensAIR ranks
    size_t model_size = sizeof(float)* 40708104;
    size_t window_size = sizeof(unsigned long int) + sizeof(int) + sizeof(int) + sizeof(int) + (sizeof(size_t)*32) + sizeof(float) + sizeof(float) + model_size;
    int dataset_size = 100000;  //tiny imagenet (number of training examples)
    int epoch_size = int(dataset_size/mini_batch_size);

    
    TensAIR::Drift_Mode drift_detector_mode=TensAIR::Drift_Mode::ALWAYS_TRAIN; //drift detector disabled 
    int print_frequency = 1;
    float convergence_factor = 1e-4;
    int epochs_for_convergence = 5;
    std::string print_to_folder = std::string(path_value) + "/output/";
    model = new VGG16_Convergence(2, rank, worldSize, window_size, broadcast_frequency, epochs, gpus_per_node, saved_model_dir, eval_data_file, "serve", epoch_size, convergence_factor, epochs_for_convergence, drift_detector_mode,print_to_folder,print_frequency); //TensAIR operator

    //link operators
    addLink(model, model);

    //init operators
    model->initialize();
}

VGG16::~VGG16() {
	delete model;
    
}
