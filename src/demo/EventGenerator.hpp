#pragma once
#include "../dataflow/BasicVertex.hpp"
#include "../communication/Message.hpp"
#include "../serialization/Serialization.hpp"
#include <string>

#include <mpi4py/mpi4py.h>
#include <deque>
#include <vector>
#include <fstream>

namespace concept_drift_cifar {

/**
 * Struct to store mini_batch data that will be serialized and sent to a TensAIR model
 * 
 * mini_batch_size is the size of the mini batch
 * num_inputs is the number of tensor inputs
 * size_inputs is a list of the size of the num_inputs inputs
 * inputs is the data of the num_inputs inputs of size = size_inputs
 */
struct Mini_Batch_Generator {
    int mini_batch_size;
    int num_inputs;
    size_t *size_inputs;
    char **inputs;
};


/// Event Generator of the concept_drift_cifar usecase
class EventGenerator: public BasicVertex<> {

public:
    
    /**
     * Default constructor
     * 
     * tag is the number of this operator in the AIR dataflow
     * rank is the number of this rank on MPI
     * worldSize is the total number of ranks on MPI
     * mini_batch_size is the size of the mini_batches generated
     * msg_sec is the throughput (number of messages processed before waiting 1sec to resume)
     * epochs is 1/2 THE NUMBER OF EPOCHS GENERATED here. epochs = the number of epochs used for training in TensAIR
     * train_data_file is the file with the training data . Format: (target context1 context2 context3 context4  context5 label1 label2 label3 label4 label5)
     * windowSize is the maximum message size received, default is 1MB
     * comm is the MPI object received when using the Python Interface
     * 
     */
    EventGenerator(const int tag, const int rank, const int worldSize, int mini_batch_size, int msg_sec, int epochs, const char* train_data_file, int windowSize = 1000000, int drift_frequency = 10, MPI_Comm comm = MPI_COMM_WORLD);

    ///Main method that manages EventGenerator dataflow
	void streamProcess(const int channel);

protected:
    
    /**
     * Generates messages based on infile file
     * 
     * quatity is the quatity of training examples per mini_batch
     * infile is the file in which the training data is stored
     */
	virtual vector<output_data> generateMessages(const unsigned int quantity, ifstream &infile);
    
    /**
     * Adds training example to Mini batch
     * 
     * micro_batch is the mini_batch being filled
     * infile is the file in which the training data is stored
     * position is the current training example being processed
     * 
     * Return: 0 = sucess; 1: epochEnd; -1: epochsEnd
     */
    int addToBatch(Mini_Batch_Generator &micro_batch, std::ifstream &infile, int position);
    
    
    
    int mini_batch_size; ///mini batch size
    int msg_sec; ///messages throughput
    int epochs; ///number of epochs processed  by TensAIR
    int epochs_generate; ///number of epochs to generate
    int starting_second = (int)MPI_Wtime(); ///time when we start to generate messages
    int inference_rank = 0; ///rank that will receive Mini batch
    vector<output_data> res; /// object that stores message and recipient
    long long int msg_count = 0; ///total number of messages generated
    int epoch = 0; ///current epoch
    string train_data_file;
    //const char* train_data_file = "../data/CIFAR/data_batch_1.bin";
    static const unsigned int event_generator_rank = 0; ///rank that will ran the EventGenerator


    int drift_frequency;
    std::vector<std::pair<int,int>> conceptDrifts;
    std::ofstream outFile;
};
};
