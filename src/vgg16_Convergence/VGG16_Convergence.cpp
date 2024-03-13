#include "VGG16_Convergence.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <vector>
#include <queue>
#include <algorithm>

VGG16_Convergence::VGG16_Convergence(const int tag, const int rank, const int worldSize, int windowSize, int broadcast_frequency, int epochs, int gpus_per_node, const char* saved_model_dir, const char* eval_data_file, const char* tags, int epoch_size, float convergence_factor, int epochs_for_convergence, TensAIR::Drift_Mode drift_detector_mode, std::string print_to_folder, int print_frequency, bool preallocate_tensors, int mini_batch_size, MPI_Comm comm) :
TensAIR(tag, rank, worldSize, windowSize, broadcast_frequency, epochs, gpus_per_node, saved_model_dir, eval_data_file, tags, epoch_size, convergence_factor, epochs_for_convergence, drift_detector_mode, print_to_folder, print_frequency, preallocate_tensors, mini_batch_size, comm) {
    this->dataset = readDataset();
    shuffleDataset();
    this->data = createMinibatches(this->dataset, mini_batch_size);
    cout << "Dataset read succesfully!" << endl;

    this->itr_per_epoch = this->data.size();

    if (!file_to_print.is_open()) {
        std::cout << "Failed to open the file." << std::endl;
        cout.flush();
        return;
    }else{
        file_to_print << "epochs," << std::to_string(this->epochs) << ",,gpusNode," << std::to_string(this->gpus_per_node) << ",,mini_batch_size," << std::to_string(mini_batch_size) << ",,rank," << std::to_string(rank) << std::endl;
        file_to_print << "Gradients applied, Gradients calculated, Loss, Time_diff(s)" << std::endl;;
        file_to_print.flush();
    }

}

void VGG16_Convergence::streamProcess(int channel) {
    MPI_Barrier(this->COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    lastUpdate = start;


    while(this->ALIVE){
        
        message_ptr message;
        //check for messages on listener
        pthread_mutex_lock(&listenerMutexes[channel]);
        bool inMessageEmpty = inMessages[channel].empty();
        if(!inMessageEmpty){
            message_ptr message2(std::move(inMessages[channel].front()));
            message = std::move(message2);
            inMessages[channel].pop_front();
        }else if(channel != rank){ //wait for new updates (if channel == rank the thread will calculate those updates)
            pthread_cond_wait(&this->listenerCondVars[channel], &this->listenerMutexes[channel]);
        }
        pthread_mutex_unlock(&listenerMutexes[channel]);
        
        
        //add received messages to update_list (if they exist)
        pthread_mutex_lock(&update_list_mutex);
        if(!inMessageEmpty){
            update_list.push_back(std::move(message));
        }
        pthread_mutex_unlock(&update_list_mutex);


        if(channel == rank){
            //get oldest message from update_list
            pthread_mutex_lock(&update_list_mutex);
            bool update = !update_list.empty();
            if(update){
                message = std::move(update_list.front());
                update_list.pop_front();
            }
            pthread_mutex_unlock(&update_list_mutex);

            
            //if no messages exists, calculate new gradient
            if(!update){ //no message received
                //create empty message to pass to processGradientCalc (minibatch size 1 and empty content)
                message_ptr message_batch = generateMessage();
                if(message_batch != NULL){
                    bool endStream = message_from_generator(std::move(message_batch));
                    if (endStream){
                        this->ALIVE=false;
                        break;
                    }
                }
            }
            
            //if there exists a message, apply its gradient
            if(update){
                auto st = std::chrono::high_resolution_clock::now();

                bool endStream = message_from_model(std::move(message));
                if (endStream){
                    this->ALIVE=false;
                    break;
                }

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double>  dur = end - st; // Calculate the time difference in microseconds
                auto sec = dur.count();
            }
        } 
    }

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = now - start; // Calculate the time difference in microseconds
    double seconds = duration.count();
    cout << std::to_string(this->gradientsApplied) << "," << std::to_string(this->gradientsCalculated) << endl;//"," << std::to_string(currentError) << endl;
    cout << "Total duration(s)," << std::to_string(seconds) << endl;
    if(file_to_print.is_open()){
        file_to_print << "Total duration(s)," << std::to_string(seconds) << endl;
        file_to_print.close();
    }
    this->ALIVE = false;
    MPI_Abort(this->COMM_WORLD, 1);
    for(int i = 0; i < previous.size()*worldSize; i ++){
        pthread_cond_signal(&this->listenerCondVars[i]);
    }
    
}

Dataset VGG16_Convergence::readDataset(){
    Dataset dataset;
    dataset.num_inputs = 0;
    
    ifstream infile(this->train_data_file); //open training data file
    int failed;
    do{
        char* img = (char*)malloc(sizeof(float)*64*64*3);
        char* label = (char*)malloc(sizeof(int));
        failed = readTrainingSample(infile, img, label);
        if (!failed){
            dataset.num_inputs++;
            dataset.imgs_labels.push_back(make_pair(img, label));
        }
    }while(!failed);
    
    infile.close();
    
    return dataset;
}

void VGG16_Convergence::shuffleDataset(){
    unsigned int seed = (10000 * rank) + epoch;
    std::mt19937 gen(seed);
    
    // Shuffle imgs and labels with the same seed
    std::shuffle(this->dataset.imgs_labels.begin(), this->dataset.imgs_labels.end(), gen);
    
    this->iterator_images_labels = this->dataset.imgs_labels.begin();
    return;
}

vector<Mini_Batch_Generator> VGG16_Convergence::createMinibatches(Dataset dataset, int mini_batch_size){
    int n_minibatches = floor(dataset.num_inputs / mini_batch_size);
    
    std::vector<Mini_Batch_Generator> mini_batches;
    
    for(int i = 0; i < n_minibatches; i++){
        Mini_Batch_Generator ubatch;
        
        ubatch.mini_batch_size=mini_batch_size;
        ubatch.num_inputs = 2;
        ubatch.size_inputs = (size_t*) malloc(sizeof(size_t)*2);
        ubatch.size_inputs[0] = sizeof(float)*mini_batch_size*64*64*3; //imgs size
        ubatch.size_inputs[1] = sizeof(int)*mini_batch_size; //labels size
        ubatch.inputs = (char**) malloc(sizeof(char*)*2);
        ubatch.inputs[0] = (char*) malloc(ubatch.size_inputs[0]); //imgs
        ubatch.inputs[1] = (char*) malloc(ubatch.size_inputs[1]); //labels
        for(int j = 0; j < mini_batch_size; j++){
            addToMiniBatch(&ubatch, j);
        }
        mini_batches.push_back(ubatch);
    }
    this->it = mini_batches.begin();
    
    return mini_batches;
}


void VGG16_Convergence::addToMiniBatch(Mini_Batch_Generator *ubatch, int position){
    pair<char*, char*> imgs_lab = *this->iterator_images_labels;
    char* img = imgs_lab.first;
    char* lab = imgs_lab.second;
    
    std::copy(static_cast<const char*>(img),
              static_cast<const char*>(img + sizeof(float)*64*64*3), &ubatch->inputs[0][64*64*3*position*sizeof(float)]);

    std::copy(static_cast<const char*>(lab),
              static_cast<const char*>(lab + sizeof(int)), &ubatch->inputs[1][position*sizeof(int)]);
    
    ++this->iterator_images_labels;
    return;
}

void VGG16_Convergence::refillMiniBatches(){
    this->it = this->data.begin();
    this->iterator_images_labels = this->dataset.imgs_labels.begin();
    
    for(int i = 0; i < this->data.size(); i++){
        for(int j = 0; j < this->data[0].mini_batch_size; j++){
            addToMiniBatch(&this->data[i], j);
        }
    }
}

Mini_Batch_Generator VGG16_Convergence::nextElement(){
    if (this->it == this->data.end()) {
        shuffleDataset();
        refillMiniBatches();
    }
    
    Mini_Batch_Generator sampledElement = *this->it;
    ++this->it;
    
    return sampledElement;
}


message_ptr VGG16_Convergence::generateMessage(){
    Mini_Batch_Generator ubatch = nextElement();
    
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
    Serialization::dynamic_event_wrap<char>(ubatch.inputs[0][0], message.get(), ubatch.size_inputs[0]); //imgs
    Serialization::dynamic_event_wrap<char>(ubatch.inputs[1][0], message.get(), ubatch.size_inputs[1]); //labels
    
    return message;
}

int VGG16_Convergence::readTrainingSample(std::ifstream &infile, char* image, char* label_from_image){
    char img[64*64*3];
    char label[1];
    
    infile.read(img, 64*64*3); // read img as char*
    infile.read(label, 1); // read label as char
    
    //convert img from char* (1 byte per pixel) to float* (4 bytes per pixel)
    float img_float[64*64*3];
    //copy float* (4 bytes per pixel) as char* (4 bytes per pixel) to img variable
    for(int i = 0; i < 64*64*3; i+=3){ // convert char to float and normalize data.
        img_float[i+0] = ((((float)(uint8_t)img[i+0])/255) - 0.48023694 ) / (0.27643643); //((byte_r / 255) - mean_r) / (std_r)
        img_float[i+1] = ((((float)(uint8_t)img[i+1])/255) - 0.44806704 ) / (0.26886328); //((byte_g / 255) - mean_g) / (std_g)
        img_float[i+2] = ((((float)(uint8_t)img[i+2])/255) - 0.39750364 ) / (0.28158993); //((byte_b / 255) - mean_b) / (std_b)
    }
    
    std::copy(static_cast<const char*>(static_cast<const void*>(img_float)),
              static_cast<const char*>(static_cast<const void*>(img_float)) + sizeof(float)*64*64*3, &image[0]);


    //convert label from char(1 byte) to int(4 bytes)
    int label_int = (int)(uint8_t)label[0];
    //copy int(4 bytes) as char* (4bytes) to label variable
    std::copy(static_cast<const char*>(static_cast<const void*>(&label_int)),
              static_cast<const char*>(static_cast<const void*>(&label_int)) + sizeof(int), &label_from_image[0]);



    //Check if the micro_batch was read successfully. If it was not succesfully is because the file ended (thus we finished the current epoch)
    if(!infile){
        return 1;
    }
    
    return 0;
}


int VGG16_Convergence::print_to_file_training(float** metrics_data, int n_metrics, int n_delta){
    
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = now - lastUpdate; // Calculate the time difference in microseconds
    double seconds = duration.count();
    lastUpdate = now;
    
    
    // Check if the file is open
    if (!this->file_to_print.is_open()) {
        std::cerr << "Error opening file to print to!" << std::endl;
        return 1;
    }

    file_to_print << gradientsApplied << "," << gradientsCalculated << "," << metrics_data[0][0]/n_delta << "," << std::to_string(seconds) << std::endl;
    cout << gradientsApplied << "," << gradientsCalculated << "," << metrics_data[0][0]/n_delta << "," << std::to_string(seconds) << std::endl;
    
    return 0;
}

bool VGG16_Convergence::end_training(float** metrics_data, int n_metrics, int n_delta){
    // maintains a sliding window of the most recent losses and their correspondent average
    for (int i = 0; i < n_delta; i++){
        recent_losses.push(metrics_data[0][0]/n_delta);
        if(recent_losses.size() > n_recent_losses){
            sum_recent_losses -= recent_losses.front();
            recent_losses.pop();
        }
    }
    sum_recent_losses += metrics_data[0][0];

    // stop training if recent average_loss is small or if maximum number of epochs reached
    if(epoch == this->epochs || sum_recent_losses < loss_objective){
            return true;
    }
    return false;
}