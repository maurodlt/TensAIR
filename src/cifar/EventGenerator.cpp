#include "EventGenerator.hpp"
#include "../usecases/Cifar.hpp"
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

using namespace cifar;

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
EventGenerator::EventGenerator(const int tag, const int rank, int worldSize, int mini_batch_size, int msg_sec, int epochs, const char* train_data_file, int windowSize,  MPI_Comm comm) :
BasicVertex<>(tag, rank, worldSize, windowSize, comm){
    this->mini_batch_size = mini_batch_size;
    this->msg_sec = msg_sec;
    this->epochs = epochs;
    this->epochs_generate = epochs*2;
    this->train_data_file = string(train_data_file);
}

///Main method that manages EventGenerator dataflow
void EventGenerator::streamProcess(const int channel){
    
    //check if current rank shall ran EventGenerator (the bottleneck is not here usually, thus a single rank is usually enough)
    if (rank == EventGenerator::event_generator_rank){
        int count = 0;
        ifstream infile(train_data_file.c_str()); //open training data file
        
        //print when started generating messages
        if (epoch == 0){
            time_t start;
            time(&start);
            cout << "Time start : " << fixed
                     << double(start);
                cout << " sec " << endl;
        }

        while(ALIVE){
            
            vector<output_data> out = generateMessages(this->mini_batch_size, infile);
            if(!out.empty()){
                send(std::move(out)); //send message to TensAIR
            }
            
            count++;
            if(count == msg_sec){ //check if we shall wait to accommodate the throughput defined
                sleep(1);
                count = 0;
             
            }
            
        }
        
    }
}


/*
 * Adds training example to Mini batch
 * 
 * micro_batch is the mini_batch being filled
 * infile is the file in which the training data is stored
 * position is the current training example being processed
 * 
 * Return: 0 = sucess; 1: epochEnd; -1: epochsEnd
 */
int EventGenerator::addToBatch(Mini_Batch_Generator &micro_batch, std::ifstream &infile, int position){
    //check if we are not overfowing the mini_batch
    if (position + 1 <= micro_batch.mini_batch_size){
        char label[1];
        char img[32*32*3];
        
        infile.read(label, 1); // read label as char
        infile.read(img, 32*32*3); // read img as char*
        
        //convert label from char(1 byte) to int(4 bytes)
        int label_int = (int)label[0];
        //copy int(4 bytes) as char* (4bytes) to mini_batch
        std::copy(static_cast<const char*>(static_cast<const void*>(&label_int)),
                  static_cast<const char*>(static_cast<const void*>(&label_int)) + sizeof(int), &micro_batch.inputs[0][position*sizeof(int)]);
        
        //convert img from char* (1 byte per pixel) to float* (4 bytes per pixel)
        float img_float[32*32*3];
        //copy float* (4 bytes per pixel) as char* (4 bytes per pixel) to mini_batch
        for(int i = 0; i < 32*32*3; i++){ // convert char to float.
            img_float[i] = ((float)(uint8_t)img[i])/255;
        }
        
        std::copy(static_cast<const char*>(static_cast<const void*>(img_float)),
                  static_cast<const char*>(static_cast<const void*>(img_float)) + sizeof(float)*32*32*3, &micro_batch.inputs[1][32*32*3*position*sizeof(float)]);
        
        //Check if the micro_batch was read successfully. If it was not succesfully is because the file ended (thus we finished the current epoch)
        if(!infile){
            infile.close();
            epoch++;
            if(epoch == epochs_generate){ //check if all epochs have being generated
                return -1;
            }
            infile.open(train_data_file.c_str()); //restart epoch
            return 1;
        }
       
    }else{
        throw "Trying to add more than mini_batch_size elements in the mini_batch;";
    }
    return 0;
}

/**
 * Generates messages based on infile file
 * 
 * quatity is the quatity of training examples per mini_batch
 * infile is the file in which the training data is stored
 */
vector<output_data> EventGenerator::generateMessages(const unsigned int quantity, ifstream &infile){
    int isLast;
    
    //init ubatch
    Mini_Batch_Generator ubatch;
    ubatch.mini_batch_size=mini_batch_size;
    ubatch.num_inputs = 2;
    ubatch.size_inputs = (size_t*) malloc(sizeof(size_t)*2); 
    ubatch.size_inputs[0] = sizeof(int)*mini_batch_size; //labels size
    ubatch.size_inputs[1] = sizeof(float)*mini_batch_size*32*32*3; //imgs size
    ubatch.inputs = (char**) malloc(sizeof(char*)*2);
    ubatch.inputs[0] = (char*) malloc(ubatch.size_inputs[0]); //labels
    ubatch.inputs[1] = (char*) malloc(ubatch.size_inputs[1]); //imgs
    
    
    //fill ubatch
    for(int i = 0; i < quantity; i++){
        isLast = addToBatch(ubatch, infile, i);
        
        if (isLast == 1){ //check if epoch ended
            vector<output_data> res_end;
            return std::move(res_end);
        }else if(isLast == -1){ //check if we already ran all epochs
            ALIVE = false;
            vector<output_data> res_end;
            return std::move(res_end);
        }
    }
    
    //calculate message size
    size_t message_size = sizeof(int) + sizeof(int) + (sizeof(size_t)*ubatch.num_inputs);
    for(int i = 0; i < ubatch.num_inputs; i++){
        message_size += ubatch.size_inputs[i];
    }
    //create message
    message_ptr message = createMessage(message_size);
    
    
    // fill message buffer
    Serialization::wrap<int>(ubatch.mini_batch_size, message.get());
    Serialization::wrap<int>(ubatch.num_inputs, message.get());
    Serialization::dynamic_event_wrap<size_t>(ubatch.size_inputs[0], message.get(), sizeof(size_t) * ubatch.num_inputs);
    Serialization::dynamic_event_wrap<char>(ubatch.inputs[0][0], message.get(), ubatch.size_inputs[0]); //labels
    Serialization::dynamic_event_wrap<char>(ubatch.inputs[1][0], message.get(), ubatch.size_inputs[1]); //imgs

    //define to which TensAIR model to send the message
    inference_rank = msg_count % worldSize;
    msg_count++;
    destination dest = vector<int>({inference_rank});
    
    vector<output_data> res;
    res.reserve(1);
    res.push_back(make_pair(std::move(message), dest));
    
    //free ubatch
    free(ubatch.size_inputs);
    for(int i = 0; i < ubatch.num_inputs; i++)
        free(ubatch.inputs[i]);
    free(ubatch.inputs);
    
        
    return res;
    
}
