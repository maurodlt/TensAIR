#pragma once
#include "BasicVertex.hpp"

#include <map>
#include <vector>
#include <mpi4py/mpi4py.h>
#include <sstream>

using std::move;
using std::make_unique;
using std::pair;
using std::cout;

/**
 * Basic implementation of information necessary
 * to process a window of events.
 * */
template<typename T>
class WindowInformation {
        
    public:
        unsigned int completeness = 0;
        vector<T> events;
        pthread_mutex_t lock;
        WindowInformation(){pthread_mutex_init(&lock, NULL);};
};

/**
 * One of the three baselines for aggregation.
 * Send an output message only when enough messages
 * have been received for the oldest window.
 * */
template<typename T>
class CountAggregator : public BasicVertex<T> {

    using Windows = map<int, WindowInformation<T>>;

    public:
        CountAggregator(const int tag, const int rank, const int worldSize);
        virtual void streamProcess(int channel);

    protected:    
        virtual vector<int> getWindowIDs(const T& event) = 0; // user must define what type of window we are dealing with
        virtual int getMaxCompleteness() = 0; // use must define the expected number of messages depending on the dataflow structure implemented
        virtual vector<output_data> processWindow(const vector<T>& events); // user can define his own version
        virtual vector<output_data> processMessage(message_ptr message); // replaces previous implementation
        virtual bool isWindowComplete(int window_id);
        virtual void updateCompleteness(int window_id);
        map<int, vector<T>> shardEvents(vector<T>& events);


    private:
        void updateEvents(int window_id, vector<T>& events);
        Windows windows;
        pthread_mutex_t windows_mtx;

};

template<typename T>
CountAggregator<T>::CountAggregator(const int tag, const int rank, const int worldSize) : 
BasicVertex<T>(tag, rank, worldSize){
    pthread_mutex_init(&windows_mtx, NULL);
}

template<typename T>
void CountAggregator<T>::streamProcess(int channel){
    BasicVertex<T>::streamProcess(channel);    
}

/**
 * Store events from every message and ask user to compute aggregate
 * when window is finished (user must also define the correct size of
 * the window and the correct way to determine the window id ; note 
 * that a message is might correpond to mutiple windows in the case
 * of sliding windows for example).
 * */
template<typename T>
vector<output_data> CountAggregator<T>::processMessage(message_ptr message){

    int message_id = BasicVertex<T>::readMessageID(message);
    char input_baseline = BasicVertex<T>::readBaseline(message);

    if(input_baseline != BasicVertex<T>::getBaseline()){
        stringstream ss;
        ss << "[CountAggregator](processMessage) Unrecognized aggregation protocol.\n"
        << "Input protocol : " << input_baseline << '\n'
        << "Expected protocol : " << BasicVertex<T>::getBaseline() << '\n';
        throw ss.str();
    }

    vector<output_data> res;
    vector<T> events = this->readEvents(message);
    map<int, vector<T>> shards = shardEvents(events);
    for (pair<const int, vector<T>>& window : shards){
        const int window_id = window.first;
        vector<T>& window_events = window.second;

        updateCompleteness(window_id);
        updateEvents(window_id, window_events);

        // output something only if the window is finished
        // do not call the original processMessage or 
        // processEvent because we don't need it
        if (isWindowComplete(window_id)){
            vector<output_data> out = move(processWindow(windows[window_id].events));

            for (output_data& data : out){
                Serialization::wrap<char>(BasicVertex<T>::getBaseline(), data.first.get());
                Serialization::wrap<int>(window_id, data.first.get());
                res.push_back(move(data));
            }
            
            pthread_mutex_lock(&windows_mtx);
            windows.erase(window_id);
            pthread_mutex_unlock(&windows_mtx);

        }
    }

    return res;
}

template<typename T>
map<int, vector<T>> CountAggregator<T>::shardEvents(vector<T>& events){
    map<int, vector<T>> res;

    for (const T& event : events){
        vector<int> window_ids = getWindowIDs(event); // implemented by user (might depend on the implementation of the event type)
        for (const int window_id : window_ids){
            res.emplace(window_id, vector<T>({}));
            res[window_id].push_back(event);
        }
    }

    return res;
}


template<typename T>
bool CountAggregator<T>::isWindowComplete(int window_id){
    return windows[window_id].completeness == getMaxCompleteness();
}

/**
 * Example of max completeness :
 * There are PER SEC MSG COUNT messages generated per second
 * There are window_duration seconds in a window
 * Each of the previous operators send a message_id once
 * Each rank sends their message to this rank
 * */
// template<typename T>
// int CountAggregator<T>::getMaxCompleteness(){
//     return window_duration * PER_SEC_MSG_COUNT * previous.size() * worldSize;
// }

/**
 * Increase the number of messages detected for the givent window.
 * */
template<typename T>
void CountAggregator<T>::updateCompleteness(int window_id){
    // update completeness information
    if (windows.count(window_id) == 0){
        pthread_mutex_lock(&windows_mtx);
        windows.emplace(window_id, WindowInformation<T>()); // initialize a new window
        pthread_mutex_unlock(&windows_mtx);
    }

    pthread_mutex_lock(&windows[window_id].lock);
    windows[window_id].completeness++; // update completeness
    pthread_mutex_unlock(&windows[window_id].lock);
}

/**
 * Stores the message events in the corresponding window storage.
 * */
template<typename T>
void CountAggregator<T>::updateEvents(int window_id, vector<T>& events){
    
    pthread_mutex_lock(&windows[window_id].lock);
    for (size_t i = 0; i < events.size(); i++){
        windows[window_id].events.push_back(events[i]); // copy
    }
    pthread_mutex_unlock(&windows[window_id].lock);
}

/**
 * User may override this method to return the result they want
 * */
template<typename T>
vector<output_data> CountAggregator<T>::processWindow(const vector<T>& events){
    message_ptr message = this->createMessage(sizeof(size_t));

    Serialization::wrap<size_t>(events.size(), message.get()); // change this if you do another implementation
    
    vector<output_data> res;
    res.push_back(make_pair(move(message), this->target_same_rank));
    return move(res);
}
