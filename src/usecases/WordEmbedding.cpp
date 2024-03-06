#include "WordEmbedding.hpp"
#include "../word_embedding/EventGenerator.hpp"
#include "../TensAIR/TensAIR.hpp"
#include "../TensAIR/DriftDetector.hpp"
#include <tensorflow/c/c_api.h>
#include <cstdlib>
#include <iostream>

using namespace word_embedding;

WordEmbedding::WordEmbedding() : Dataflow() {
    int mini_batch_size = 128;
    int epochs = 300;
    int broadcast_frequency = 1; //mini batches per broadcast (recommended to set as the world_size)
    int gpus_per_node = 0;
    int throughput = 500;//messages in a row before waiting 1 sec
    char* path_value = std::getenv("TENSAIR_PATH");
    std::string train_data_file_str = std::string(path_value) + "/data/w2v/shakespeare_train.txt"; //Created using W2V_data notebook
    const char* train_data_file = train_data_file_str.c_str();
    generator = new word_embedding::EventGenerator(1, rank, worldSize, mini_batch_size, throughput, epochs, train_data_file); //Event Generator operator
    
    std::string saved_model_dir_str = std::string(path_value) + "/data/w2v/w2v_model.tf"; //file with tf model created using W2V-Model notebook
    const char* saved_model_dir = saved_model_dir_str.c_str();
    const char* eval_data_file = ""; //#available under /data/w2v/shakespeare_eval.bytes (Created using W2V_data notebook)
    //max message size involving TensAIR
    int window_size = sizeof(unsigned long int) + sizeof(int) + sizeof(int) + sizeof(int) + (sizeof(size_t)*4) + sizeof(float) + sizeof(float) + (sizeof(float)*50000*300*2);
    int dataset_size = 112640;  //shakespeare
    int epoch_size = int(dataset_size/mini_batch_size); //number of mini batches per epoch
    TensAIR::Drift_Mode drift_detector_mode = TensAIR::Drift_Mode::ALWAYS_TRAIN; //drift detector disabled 
    string print_to_folder = "";
    int print_frequency = 1;
    float convergence_factor = 1e-4;
    int epochs_for_convergence = 5;
    model = new TensAIR(2, rank, worldSize, window_size, broadcast_frequency, epochs, gpus_per_node, saved_model_dir, eval_data_file, "serve",epoch_size,convergence_factor,epochs_for_convergence,drift_detector_mode,print_to_folder,print_frequency); //TensAIR operator
    
    //link operators
    addLink(generator, model);
    addLink(model, model);
    
    //init operators
    generator->initialize();
    model->initialize();
    
}

WordEmbedding::~WordEmbedding() {

	delete generator;
	delete model;
    
}
