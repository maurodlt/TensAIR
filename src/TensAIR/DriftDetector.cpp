#include "DriftDetector.hpp"
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
#include <deque>
#include <numeric>
#include <cmath>
#include <iostream>

using namespace drift_detector;

DriftDetector::DriftDetector(const int tag, const int rank, int worldSize, int windowSize, int max_widowLoss, string file_cuts, MPI_Comm comm) :
BasicVertex<>(tag, rank, worldSize, windowSize, comm), W(max_widowLoss){
    this->max_widowLoss = max_widowLoss;
    this->file_cuts = file_cuts;

    if(rank == DriftDetector::drift_detector_rank){
        readCuts(file_cuts);
    }
}

///Main method that manages EventGenerator dataflow
void DriftDetector::streamProcess(const int channel){
    
    //check if current rank shall ran EventGenerator (the bottleneck is not here usually, thus a single rank is usually enough)
    if (rank == DriftDetector::drift_detector_rank){
        
        while(ALIVE){
            //update drift_list
            fetchUpdateMessages(channel, message_list);

            //one thread applies the updates
            if(channel == DriftDetector::drift_detector_rank){
                message_ptr message = fetchNextMessage(channel, message_list);
                float current_loss = readLoss(std::move(message));
                
                //addLoss(current_loss);

                //float drift_magnetude = identifyDrift(); //drift_magnetude = -1 if nothing changed; 0 if model converged; float if drift ocurred (magnetude of drift)
                float drift_magnetude;
                
                if(update(current_loss)){
                    drift_magnetude=1;
                }else{
                    drift_magnetude= 0;
                }

                //if drift occurred
                if(drift_magnetude != 0){
                    vector<output_data> out = produceMessage(drift_magnetude);
                    this->send(std::move(out));
                    
                    currentLoss.clear();
                    historicalLoss.clear();
                    mean_historicalLoss = 1000000000.0;
                    sd_historicalLoss = 0.0;
                }
                
            }
        }
    }
}

void DriftDetector::fetchUpdateMessages(int channel, list<message_ptr>& pthread_waiting_list){
    pthread_mutex_lock(&this->listenerMutexes[channel]);
    
    //wait until new messages arrive in this or in other channels
    if(channel == drift_detector_rank){
        while(this->inMessages[channel].empty() && pthread_waiting_list.empty()){
            pthread_cond_signal(&empty_list_cond); 
            pthread_cond_wait(&this->listenerCondVars[channel], &this->listenerMutexes[channel]);
        }
    }else{// wait for messages in this channel
        while(this->inMessages[channel].empty()){
            pthread_cond_signal(&empty_list_cond); //signal that the drift detector has no new losses yet.
            pthread_cond_wait(&this->listenerCondVars[channel], &this->listenerMutexes[channel]);
        }
    }
    
    // when new messages arrive, add them to the list
    if(!this->inMessages[channel].empty()){
        
        pthread_mutex_lock(&update_list_mutex);
        while(!this->inMessages[channel].empty()){
            message_ptr inMessage(this->inMessages[channel].front());
            pthread_waiting_list.push_back(std::move(inMessage));
            this->inMessages[channel].pop_front();
        }
        pthread_mutex_unlock(&update_list_mutex);
        pthread_cond_signal(&this->listenerCondVars[drift_detector_rank]); //signal to drift_detector_rank channel that new messages arrived
    }
    pthread_mutex_unlock(&this->listenerMutexes[channel]);
    return;
}

//fetch messages from list buffer
message_ptr DriftDetector::fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list){
    pthread_mutex_lock(&update_list_mutex);

    message_ptr message = std::move(pthread_waiting_list.front()); // move
    pthread_waiting_list.pop_front();
    
    pthread_mutex_unlock(&update_list_mutex);

    return message;
}

float DriftDetector::readLoss(message_ptr message){
    int offset = 0;
    float loss = Serialization::read_front<float>(message.get(), offset); //read loss
    offset += sizeof(float);
    int reset_detector = Serialization::read_front<int>(message.get(), offset); //read loss
    if(reset_detector == 1){
        drift_reaction();
    }


    return loss;
}


void DriftDetector::addLoss(float loss){
    //add new loss
    currentLoss.push_back(loss);
    if (currentLoss.size() > max_widowLoss){
        //move historical sliding window 
        historicalLoss.push_back(currentLoss[0]);

        //remove old elements from sliding windows
        currentLoss.erase(currentLoss.begin());
        if(historicalLoss.size() > max_widowLoss){
            historicalLoss.erase(historicalLoss.begin());
        }
        //update historical loss metrics
        pair<float,float> metrics = mean_and_std_deviation(historicalLoss);
        mean_historicalLoss = metrics.first;
        sd_historicalLoss = metrics.second;
    }
    return;
}

void DriftDetector::insert_to_W(float loss){
    float popped = this->W.add(loss);
    add_running_stdev(false, std::vector<float>{loss});

    if(popped != -1){ // list is full
        //remove excedent value from running stdev
        pop_from_running_stdev(true, std::vector<float>{popped});

        //walk with sliding window
        pop_from_running_stdev(false, std::vector<float>{W.get(last_opt_cut)});
        add_running_stdev(true, std::vector<float>{W.get(last_opt_cut)});
    }

    this->itt++;
    
    return;
}


void DriftDetector::add_running_stdev(bool historical, vector<float> x){
    if(historical){
        for(int i = 0; i < x.size(); i++){
            summation_h += x[i];
            S_h += x[i]*x[i];
        }
        count_h += x.size();

        if (count_h > 1 && S_h > 0){
            stdev_h = sqrt((count_h * S_h)-(summation_h*summation_h)) / count_h;
        }else{
            stdev_h = 0;
        }
    }else{
        for(int i = 0; i < x.size(); i++){
            summation_new += x[i];
            S_new += x[i]*x[i];
        }
        count_new += x.size();
        
        if (count_new > 1 && S_new > 0){
            stdev_new = sqrt((count_new * S_new)-(summation_new*summation_new)) / count_new;
        }else{
            stdev_new = 0;
        }
    }
}

void DriftDetector::pop_from_running_stdev(bool historical, vector<float> x){
    if(historical){
        for(int i = 0; i < x.size(); i++){
            summation_h -= x[i];
            S_h -= x[i]*x[i];
        }
        count_h -= x.size();

        if(count_h > 1 && S_h > 0){
            stdev_h = sqrt((count_h * S_h)-(summation_h*summation_h)) / count_h;
        }else{
            stdev_h = 0;
        }
    }else{
        for(int i = 0; i < x.size(); i++){
            summation_new -= x[i];
            S_new -= x[i]*x[i];
        }
        count_new -= x.size();

        if(count_new > 1 && S_new > 0){
            stdev_new = sqrt((count_new * S_new)-(summation_new*summation_new)) / count_new;
        }else{
            stdev_new = 0;
        }
    }

}


bool DriftDetector::update(float x){
    //add new element to window
    iteration++;
    insert_to_W(x);
    delay = 0;


    //check if window is too small
    if(W.length < min_widowLoss){
        return false;
    }

    int optimal_cut = opt_cut[W.length];
    float phi_opt = opt_phi[W.length];

    
    //update running stdev and avg
    if(optimal_cut > last_opt_cut){ //remove elements from window_new and add them to window_h
        pop_from_running_stdev(false, W.getInterval(last_opt_cut, optimal_cut));
        add_running_stdev(true, W.getInterval(last_opt_cut,optimal_cut));
    }else{ //remove elements from window_h and add them to window_new
        pop_from_running_stdev(true, W.getInterval(optimal_cut, last_opt_cut));
        add_running_stdev(false, W.getInterval(optimal_cut, last_opt_cut));
    }

   
    float avg_h = summation_h / count_h;
    float avg_new = summation_new / count_new;
    stdev_h = sqrt((count_h * S_h)-(summation_h*summation_h)) / count_h;
    stdev_new = sqrt((count_new * S_new)-(summation_new*summation_new)) / count_new;

    last_opt_cut = optimal_cut;

    //add minimal noise to stdev
    stdev_h += minimum_noise;
    stdev_new += minimum_noise;

    //check t-stat
    float t_stat_value = t_stats[W.length];

    //t-test
    float t_test_result = (avg_new-avg_h) / (sqrt((stdev_new/(W.length-optimal_cut))+(stdev_h/optimal_cut)));
    if(t_test_result > t_stat_value){
        drift_reaction();
        return true;
    }

    if (((stdev_new*stdev_new)/(stdev_h*stdev_h)) > phi_opt){
        if(avg_h < avg_new){
            drift_reaction();
            return true;
        }
    }
    
    return false;

}

void DriftDetector::drift_reaction(){
    W.reset();
    stdev_new = 0;
    summation_new = 0;
    count_new = 0;
    S_new = 0;
    stdev_h = 0;
    summation_h = 0;
    count_h = 0;
    S_h = 0;
    last_opt_cut = 0;
}


//return
// 0 no drift; 
// float if drift ocurred (magnetude of drift)
float DriftDetector::identifyDrift(){
    //calculate mean and std_deviation
    pair<float,float> metrics = mean_and_std_deviation(currentLoss);
    float mean_currentLoss = metrics.first;
    float sd_currentLoss = metrics.second;

    //prediction in progress
    // drift in the mean
    if(mean_currentLoss > mean_historicalLoss + 3*sd_historicalLoss){
        return mean_currentLoss / mean_historicalLoss;
    }
    
    return 0;
}


bool DriftDetector::model_converged(){

    return false;
}

void DriftDetector::adapt_historicalLoss(){
    //calculate mean and std_deviation
    pair<float,float> metrics = mean_and_std_deviation(historicalLoss);
    mean_historicalLoss = metrics.first;
    sd_historicalLoss = metrics.second;

    historicalLoss.clear();
    historicalLoss.assign(currentLoss.begin(), currentLoss.end());
    currentLoss.clear();
    return;
}

pair<float,float> DriftDetector::mean_and_std_deviation(vector<float> series){
    //calculates mean
    float sum = std::accumulate(series.begin(), series.end(), 0.0);
    float mean = sum / series.size();

    //calculates std_deviation
    std::vector<float> diff(series.size());
    std::transform(series.begin(), series.end(), diff.begin(), [mean](float x) { return x - mean; });
    float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float std_deviation = std::sqrt(sq_sum / series.size());

    return make_pair(mean, std_deviation);
}

vector<output_data> DriftDetector::produceMessage(float drift_magnetude){
    message_ptr message = createMessage(sizeof(float)); //allocate memory for message

    Serialization::wrap<float>(drift_magnetude, message.get()); 

    destination dest = vector<int>({target_all_ranks});
    vector<output_data> res;
    res.push_back(make_pair(std::move(message), dest));

    return res;
}

int DriftDetector::readCuts(string filename){
    ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    string line;

    std::getline(file, line);
    std::stringstream lineStream(line);
    std::string cell;


    getline(lineStream, cell, ',');
    float confidence = std::stof(cell);
    getline(lineStream, cell, ',');
    float rho = std::stof(cell);
    getline(lineStream, cell, ',');
    int max_window_size = std::stoi(cell);
    getline(lineStream, cell, ',');
    int min_window_size = std::stoi(cell);
    getline(lineStream, cell, ',');

    for(int i = 0; i < max_window_size+2; i++){
        getline(lineStream, cell, ',');
        opt_cut.push_back(std::stoi(cell));
    }

    for(int i = 0; i < max_window_size+2; i++){
        getline(lineStream, cell, ',');
        opt_phi.push_back(std::stof(cell));
    }

    for(int i = 0; i < max_window_size+2; i++){
        getline(lineStream, cell, ',');
        t_stats.push_back(std::stof(cell));
    }

    for(int i = 0; i < max_window_size+2; i++){
        getline(lineStream, cell, ',');
        t_warning.push_back(std::stof(cell));
    }

    file.close();
    return 0;
}




