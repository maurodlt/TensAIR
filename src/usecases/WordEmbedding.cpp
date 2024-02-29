#include "WordEmbedding.hpp"
#include "../word_embedding/EventGenerator.hpp"
#include "../TensAIR/TensAIR.hpp"
#include "../TensAIR/DriftDetector.hpp"
#include <tensorflow/c/c_api.h>

using namespace word_embedding;

WordEmbedding::WordEmbedding() : Dataflow() {
    int mini_batch_size = 32;
    int epochs = 10;
    int broadcast_frequency = 1; //mini batches per message (IN DEVELOPMENT, USE 1)
    int gpus_per_node = 0;
    int throughput = 10000;//messages in a row before waiting 1 sec
    const char* train_data_file = "../data/W2V/wikipedia1-train.txt"; //file with trining data
    generator = new word_embedding::EventGenerator(1, rank, worldSize, mini_batch_size, throughput, epochs, train_data_file); //Event Generator operator
    
    const char* saved_model_dir = "../data/W2V/python_interface/w2v_shakespeare.tf"; //file with tf model
    const char* eval_data_file = "../data/W2V/wikipedia1-evaluate.bytes"; //file with evaluation data (parsed to binary file)
    int train_examples = 5; //contexts per training example (true context and negative samples)
    int embedding_size = 300; //embedding size of NN model
    //max message size involving TensAIR
    int window_size = sizeof(unsigned long int) + sizeof(int) + sizeof(int) + sizeof(int) + (sizeof(size_t)*6) + sizeof(float) + sizeof(float) + (sizeof(int)*mini_batch_size) + (sizeof(float)*mini_batch_size*300) + (sizeof(int) * mini_batch_size * 5) + (sizeof(float)*mini_batch_size*5*300);

    int dataset_size = 11793644;   //1% wikipedia
    //int dataset_size = 112640;  //shakespeare
    int epoch_size = int(dataset_size/mini_batch_size); //number of mini batches per epoch
    TensAIR::Drift_Mode drift_detector_mode = TensAIR::Drift_Mode::ALWAYS_TRAIN; //drift detector enabled
    string print_to_folder = "../output/";
    int print_frequency = 10;
    float convergence_factor = 1e-2;
    int epochs_for_convergence = 2;
    model = new TensAIR(2, rank, worldSize, window_size, broadcast_frequency, epochs, gpus_per_node, saved_model_dir, eval_data_file, "serve",epoch_size,convergence_factor,epochs_for_convergence,drift_detector_mode,print_to_folder,print_frequency); //TensAIR operator
    //fix dimentions that could not be retrieved successfuly with saved_model_cli
    TensAIR* model1 = dynamic_cast<TensAIR*>(model);
    model1->retrieve_delta_output_dims = {{1},{1},{-1},{-1,300},{-1*5},{-1*5,300}};
    //model1->gradient_calc_output_dims = {{1},{1},{-1},{-1,300},{-1*5},{-1*5,300}};
    //model1->apply_gradient_input_dims = {{-1},{-1,300},{-1*5},{-1*5,300}};
    drift_detector = new drift_detector::DriftDetector(3, rank, worldSize);
    
    //link operators
    addLink(generator, model1);
    addLink(model1, model1);
    addLink(model1, drift_detector);
    addLink(drift_detector, model1);

    //init operators
    generator->initialize();
    model1->initialize();
    drift_detector->initialize();
    
}


WordEmbedding::~WordEmbedding() {

	delete generator;
	delete model;
    
}
