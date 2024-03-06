#pragma once
#include "../dataflow/BasicVertex.hpp"
#include "../communication/Message.hpp"
#include "../serialization/Serialization.hpp"
#include <string>
#include <fstream>

#include <mpi4py/mpi4py.h>
#include <deque>

namespace event_generator {

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


/// Event Generator of General python usecase
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
     * windowSize is the maximum message size received, default is 1MB
     * init_code_file which defines method next_message(mini_batch_size)
     * comm is the MPI object received when using the Python Interface
     * 
     */
    EventGenerator(const int tag, const int rank, const int worldSize, int mini_batch_size, int msg_sec, int windowSize = 1000000, string init_code_file = "", MPI_Comm comm = MPI_COMM_WORLD);

    ///Main method that manages EventGenerator dataflow
	void streamProcess(const int channel);

protected:
    
    /**
     * Generates messages based on infile file
     * 
     * quatity is the quatity of training examples per mini_batch
     * infile is the file in which the training data is stored
     */
	virtual vector<output_data> generateMessages();
    std::string readFile(const std::string& fileName);
    
    int mini_batch_size; ///mini batch size
    int msg_sec; ///messages throughput
    int starting_second = (int)MPI_Wtime(); ///time when we start to generate messages
    int inference_rank = 0; ///rank that will receive Mini batch
    vector<output_data> res; /// object that stores message and recipient
    long long int msg_count = 0; ///total number of messages generated
    unsigned int event_generator_rank = 0; ///rank that will ran the EventGenerator
    std::string init_code_file="";
};
};
