//#include "EventGenerator.hpp"
//#include "../nexmark_gen/Random.hpp"
//#include "../usecases/Hessian_W2V.hpp"
//#include <time.h>       /* time */
//#include <unistd.h> // usleep
//#include <set> //test
//#include <sstream>
//#include <algorithm>
//#include <iterator>
//#include <string>
//#include <string.h>
//#include <ctime>
//#include <fstream>
//#include <ctime>
//
//using namespace hessian_w2v;
//using nexmark_gen::Random;
//
///**
// * Simple constructor to initialize the basic vertex values.
// * 
// * Throughput doesn't really matter because we are only trying
// * to debug the different baselines for now.
// * */
//
//EventGenerator::EventGenerator(const int tag, const int rank, int worldSize, int mini_batch_size, int msg_sec, int epochs) :
//BasicVertex<>(tag, rank, worldSize){
//    setBaseline('t'); //train baseline
//    increaseHeaderSize(sizeof(unsigned long int));
//    increaseHeaderSize(sizeof(float));
//    this->mini_batch_size = mini_batch_size;
//    this->msg_sec = msg_sec;
//    this->epochs = epochs;
//    this->epochs_generate = epochs*2;
//}
//
//
//void EventGenerator::streamProcess(const int channel){
//    
//    if (rank == Hessian_W2V::event_generator_rank){
//        //sleep(8);
//        int count = 0;
//        ifstream infile(dataset_file);
//        
//        if (epoch == 0){
//            time_t start;
//            time(&start);
//            cout << "Time start : " << fixed
//                     << double(start);
//                cout << " sec " << endl;
//        }
//        while(ALIVE){
//            
//            vector<output_data> out = generateMessages(this->mini_batch_size, infile);
//            if(!out.empty()){
//                send(move(out));
//            }else{
//                if(this->ALIVE == false){
//                    this->ALIVE = true;
//                    return;
//                }
//            }
//            //sleep(1);
//            
//            
//            
//            count++;
//            if(count == msg_sec){
//                sleep(1);
//                count = 0;
//                //send(move(this->res));
//                //break;
//            }
//             
//            //sleep(1);
//            //sleep((rand()%2) + 1);
//            
//        }
//        
//        
//    }
//}
//
//
//// 0: sucess
//// 1: epochEnd
//// -1: epochsEnd
//int EventGenerator::addToBatch(Mini_Batch_old &micro_batch, std::ifstream &infile){
//    if (micro_batch.position + 1 <= micro_batch.size){
//        
//        int position = micro_batch.position;
//        
//        infile >> micro_batch.target[position];
//        
//        for (int i = 0; i < TRAIN_EXAMPLES; i++){
//            infile >> micro_batch.context[(position*TRAIN_EXAMPLES)+i];
//            //micro_batch.context[(position*TRAIN_EXAMPLES)+i] = 123;
//        }
//        for (int i = 0; i < TRAIN_EXAMPLES; i++){
//            infile >> micro_batch.label[(position*TRAIN_EXAMPLES) + i];
//            //micro_batch.label[(position*TRAIN_EXAMPLES) + i] = 0;
//        }
//        
//        //Check if the micro_batch was read successfully
//        if(!infile){
//            //cout << "\n\n ------ Error reading file --------- \nEither the file ended or there is a wrongly formatted entry.\n\n";
//            infile.close();
//            epoch++;
//            if(epoch == this->epochs_generate){
//                return -1;
//                //sleep(1000005);
//            }
//            infile.open(dataset_file); //restart epoch
//            return 1;
//        }
//        
//            micro_batch.position++;
//       
//    }else{
//        throw "Trying to add more than mini_batch_size elements in the mini_batch;";
//    }
//    return 0;
//}
//
///**
// * New method to generate random messages.
// * 
// * - quantity is the number of messages we want to generate.
// * */
//vector<output_data> EventGenerator::generateMessages(const unsigned int quantity, ifstream &infile){
//    
//    unsigned int message_id = (iteration) * worldSize + rank;
//    int isLast = 0;
//    
//    size_t ubatch_memory_size = 0;
//    
//    //init ubatch
//    Mini_Batch_old ubatch;
//    ubatch.size = mini_batch_size;
//    ubatch.target = (int*) malloc(sizeof(int)*mini_batch_size);
//    ubatch.label = (int*) malloc(sizeof(int)*mini_batch_size*TRAIN_EXAMPLES);
//    ubatch.context = (int*) malloc(sizeof(int)*mini_batch_size*TRAIN_EXAMPLES);
//    ubatch_memory_size = sizeof(Mini_Batch_old) + ( mini_batch_size * sizeof(int) ) + (mini_batch_size * TRAIN_EXAMPLES * sizeof(int) * 2);
//    
//    
//    //fill ubatch
//    for(int i = 0; i < quantity; i++){
//        isLast = addToBatch(ubatch, infile);
//        
//        if (isLast == 1){ //endOfEpoch
//            vector<output_data> res_end;
//            return move(res_end);
//        }else if(isLast == -1){ //endOfEpochs
//            ALIVE = false;
//            vector<output_data> res_end;
//            return move(res_end);
//        }
//    }
//    
//    message_ptr message = createMessage(ubatch_memory_size);
//    
//    
//    // fill message buffer
//    //Serialization::wrap<Train_Examples>(train_examples, message.get());
//    Serialization::wrap<Mini_Batch_old>(ubatch, message.get());
//    Serialization::dynamic_event_wrap<int>(ubatch.target[0], message.get(), sizeof(int) * mini_batch_size);
//    Serialization::dynamic_event_wrap<int>(ubatch.label[0], message.get(), sizeof(int) * TRAIN_EXAMPLES * mini_batch_size);
//    Serialization::dynamic_event_wrap<int>(ubatch.context[0], message.get(), sizeof(int) * TRAIN_EXAMPLES * mini_batch_size);
//    
//    
//    
//    // add 0, baseline and message_id
//    Serialization::wrap<float>(0.0, message.get());
//    Serialization::wrap<unsigned long int>(0, message.get());
//    Serialization::wrap<char>(getBaseline(), message.get());
//    Serialization::wrap<int>(message_id, message.get());
//    
//
//    // add message to output
//    //inference_rank = ubatch.context[2] % (worldSize);
//    inference_rank = msg_count % worldSize;
//    msg_count++;
//    destination dest = vector<int>({inference_rank});
//    
//    vector<output_data> res;
//    res.reserve(1);
//    res.push_back(make_pair(move(message), dest));
//    
//    free(ubatch.target);
//    free(ubatch.context);
//    free(ubatch.label);
//    
//        
//    return res;
//    
//}
//
///**
// * Decorate the send method to update the iteration value
// * after all the current iteration's messages are sent
// * */
//void EventGenerator::send(vector<output_data> messages){
//	BasicVertex<>::send(move(messages));
//	iteration++;
//}
