#include "EventGenerator.hpp"
#include <time.h>       /* time */
#include <unistd.h> // usleep
#include <set> //test
#include <sstream>
#include <algorithm>
#include <iterator>
#include <string>
#include <string.h>
#include <ctime>
#include <fstream>
#include <ctime>
//#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // for the Python interpreter
#include <chrono>

using namespace event_generator;

namespace py = pybind11;

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
EventGenerator::EventGenerator(const int tag, const int rank, int worldSize, int mini_batch_size, int msg_sec, int windowSize,  string init_code_file, MPI_Comm comm) :
BasicVertex<>(tag, rank, worldSize, windowSize, comm){
    this->mini_batch_size = mini_batch_size;
    this->msg_sec = msg_sec;

    outFile.open("/Users/mauro.dalleluccatosi/Documents/GitHub/TensAIR/output/out.txt");

    if (rank == EventGenerator::event_generator_rank){
        this->init_code_file = init_code_file;
            if(!this->init_code_file.empty()){
                std::string init_code = readFile(init_code_file);
                py::exec(init_code);
            }
    }
}

std::string EventGenerator::readFile(const std::string& fileName) {
    std::ifstream file(fileName);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

///Main method that manages EventGenerator dataflow
void EventGenerator::streamProcess(const int channel){
    
    //check if current rank shall ran EventGenerator (the bottleneck is not here usually, thus a single rank is usually enough)
    if (rank == EventGenerator::event_generator_rank){
        int count = 0;

        while(ALIVE){
            
            vector<output_data> out = generateMessages();
            if(!out.empty()){
                send(move(out)); //send message to TensAIR
            }
            
            count++;
            if(count == msg_sec){ //check if we shall wait to accommodate the throughput defined
                sleep(1);
                count = 0;
             
            }
            
        }
        
    }
}

/**
 * Generates messages
 * 
 * quatity is the quatity of training examples per mini_batch
 */
vector<output_data> EventGenerator::generateMessages(){
    auto start = std::chrono::high_resolution_clock::now();
    // Evaluate an isolated expression
    py::bytes bytes_result = py::eval("next_message("+std::to_string(this->mini_batch_size)+")").cast<py::bytes>();

    // Get the size of the message
    size_t message_size = PyBytes_Size(bytes_result.ptr());

    message_ptr message = createMessage(message_size);

    char* msg_content = PyBytes_AsString(bytes_result.ptr());

    // serialize message
    Serialization::dynamic_event_wrap<char>(msg_content[0], message.get(), message_size); //imgs


    //define to which TensAIR model to send the message
    inference_rank = msg_count % worldSize;
    msg_count++;
    destination dest = vector<int>({inference_rank});
    outFile << inference_rank << endl;
    
    vector<output_data> res;
    res.push_back(make_pair(move(message), dest));

    return res;
}
