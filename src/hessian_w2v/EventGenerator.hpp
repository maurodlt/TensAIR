//#pragma once
//#include "../dataflow/BasicVertex.hpp"
//#include "../communication/Message.hpp"
//#include "../serialization/Serialization.hpp"
//#include <string>
//
//#include <mpi4py/mpi4py.h>
//#include <deque>
//
//#define TRAIN_EXAMPLES 5
//
//#define EMBEDDING_SIZE 300
//
////#define EXAMPLES_EPOCH 1765    //32 batch
////#define EXAMPLES_EPOCH 441     //128 batch
////#define EXAMPLES_EPOCH 220     //512 batch
////#define EXAMPLES_EPOCH 27      //2048 batch
////#define DS_SIZE 11793644   //1% wikipedia
//#define DS_SIZE 112640  //shakespeare
////#define EPOCHS 200//epochs generated
////#define EPOCHS_STOP 100 //epochs trained
//
//namespace hessian_w2v { // add class to benchmark namespace
//
//
//struct Mini_Batch_old {
//    int size;
//    float loss = 0.0;
//    int position = 0;
//    int *target;
//    int *label;
//    int *context;
//};
//
//
///**
// * Uses the default char typename, but anyway we won't read events
// * so the input's typename could be anything really, and it wouldn't
// * really matter.
// * */
//class EventGenerator: public BasicVertex<> {
//
//public:
//    
//    EventGenerator(const int tag, const int rank, const int worldSize, int mini_batch_size, int msg_sec, int epochs);
//
//	void streamProcess(const int channel);
//
//protected:
//
//	virtual vector<output_data> generateMessages(const unsigned int quantity, ifstream &infile);
//	virtual void send(vector<output_data> messages);
//    
//    int addToBatch(Mini_Batch_old &micro_batch, std::ifstream &infile);
//    
//    // generators only have one thread per rank so we can store some values here
//    int mini_batch_size;
//    int msg_sec;
//    int epochs;
//    int epochs_generate;
//    int iteration = 0;
//    int starting_second = (int)MPI_Wtime();
//    unsigned int window_duration = 1;
//    int inference_rank = 0;
//    vector<output_data> res;
//    long long int msg_count = 0;
//    int count = 0;
//    int epoch = 0;
//    const char* dataset_file = "../data/W2V/word_embedding_shakespeare_train.txt";
//    //const char* dataset_file = "../data/W2V/wikipedia1-train.txt";
//
//};
//};
