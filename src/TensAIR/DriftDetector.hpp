//IMPLEMENTATION OF OPTWIN DRIFT DETECTOR
//https://github.com/maurodlt/OPTWIN

#pragma once
#include "../dataflow/BasicVertex.hpp"
#include "../communication/Message.hpp"
#include "../serialization/Serialization.hpp"
#include "CircularList.cpp"
#include <mpi4py/mpi4py.h>

namespace drift_detector { 

class DriftDetector: public BasicVertex<> {

public:
    DriftDetector(const int tag, const int rank, const int worldSize, int windowSize = sizeof(int) + sizeof(float), int max_widowLoss = 1000, string file_cuts = "", MPI_Comm comm = MPI_COMM_WORLD);

    ///Main method that manages the Drift Detector dataflow
	void streamProcess(const int channel);

protected:
	float readLoss(message_ptr message);
    float identifyDrift();
    vector<output_data> produceMessage(float drift_magnetude);
    void adapt_historicalLoss();
    void addLoss(float loss);
    pair<float,float> mean_and_std_deviation(vector<float> series);
    bool model_converged();
    void insert_to_W(float loss);
    void add_running_stdev(bool historical, vector<float> x);
    void pop_from_running_stdev(bool historical, vector<float> x);
    bool update(float x);
    void drift_reaction();
    int readCuts(string filename);


    void fetchUpdateMessages(int channel, list<message_ptr>& pthread_waiting_list);
    message_ptr fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list);
    pthread_mutex_t empty_list_mutex = PTHREAD_MUTEX_INITIALIZER; ///mutex that identifies if evaluation is currently in course (cannot train during evaluation)
    pthread_cond_t empty_list_cond = PTHREAD_COND_INITIALIZER; ///signale mutex that identifies if evaluation is currently in course (cannot train during evaluation)
    pthread_mutex_t update_list_mutex = PTHREAD_MUTEX_INITIALIZER; ///mutex that identifies if the incoming messages buffer is being accessed
    pthread_cond_t update_list_cond = PTHREAD_COND_INITIALIZER; ///signal to mutex that identifies if the incoming messages buffer is being accessed
        

    list<message_ptr> message_list; ///buffer of messages received from Drift Detector

    //store historical loss data
    vector<float> historicalLoss;
    float mean_historicalLoss = 1000000000.0;
    float sd_historicalLoss = 0.0;

    //store current loss data
    vector<float> currentLoss;

    bool checkConvergence = false;
    bool checkDrift = false;

    int currentDrift = 1;

    int drift_detector_rank = 0;
    int max_widowLoss;

    //std::deque<float> W_new;
    //std::deque<float> W_hist;
    //std::deque<float> W;
    CircularList W;

    //Running stdev and avg
    float stdev_new = 0;
    float summation_new = 0;
    int count_new = 0;
    float S_new = 0;
    float stdev_h = 0;
    float summation_h = 0;
    int count_h = 0;
    float S_h = 0;
    int itt = 0;

    int last_opt_cut;
    int iteration = 0;
    int delay = 0;
    int min_widowLoss = 30;

    vector<int> opt_cut;
    vector<float> opt_phi;
    vector<float> t_stats;
    vector<float> t_warning;
    string file_cuts;


    float minimum_noise = 1e-6;
};
};
