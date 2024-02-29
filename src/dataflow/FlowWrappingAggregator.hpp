#pragma once
#include "CountAggregator.hpp"

/**
 * Extending the base information of a window
 * to add the max completeness value.
 * */
template<typename T>
class WindowInformationFW : public WindowInformation<T> {
            
    public:
        unsigned int max_completeness = 1;
        WindowInformationFW() : WindowInformation<T>() {};
};

/**
 * This is a subclass of the count aggregator that processes
 * windows with wrapper units.
 * */
template<typename T>
class FlowWrappingAggregator : public CountAggregator<T> {

    using WindowsFW = map<int, WindowInformationFW<T>>;

    public:
        FlowWrappingAggregator(const int tag, const int rank, const int worldSize);

    protected:
        virtual vector<output_data> processMessage(message_ptr message);
        bool isWindowComplete(int window_id);
        void updateFlowCompleteness(const WrapperUnit& unit);
        vector<WrapperUnit> extractWrapperUnits(const message_ptr& message);

    private:
        WindowsFW windows_fw;
        pthread_mutex_t windows_mtx;
};

template<typename T>
FlowWrappingAggregator<T>::FlowWrappingAggregator(const int tag, const int rank, const int worldSize):
CountAggregator<T>(tag, rank, worldSize)
{
    pthread_mutex_init(&windows_mtx, NULL);
}

/**
 * Overloading the message processing method allows us to process
 * the wrapper units before processing the messages normally.
 * */
template<typename T>
vector<output_data> FlowWrappingAggregator<T>::processMessage(message_ptr message){
    vector<WrapperUnit> wrapper_units = extractWrapperUnits(message);

    pthread_mutex_lock(&windows_mtx);

    // update before calling count aggregator's processMessage() method
    // because it will call isWindowComplete to make sure the window is 
    // finished and only the flow wrapping aggregator's completeness
    // value is taken into account, so it must be up to date at that
    // moment.
    vector<long int> windows_to_remove;
    for (const WrapperUnit& unit : wrapper_units){
        updateFlowCompleteness(unit);

        if (isWindowComplete(unit.window_start_time)) {
            windows_to_remove.push_back(unit.window_start_time);
        }
    }

    // shards events and everything...
    // note : outputs a result if completeness is at max value
    vector<output_data> res = CountAggregator<T>::processMessage(move(message));

    // remove complete windows once result has been outputed
    for (auto const& window_id : windows_to_remove) {
        windows_fw.erase(window_id);
    }
    
    pthread_mutex_unlock(&windows_mtx);

    return res;
}

/**
 * Updates numerator and denominator for flow-wrapping completeness
 * computation. 
 * */
template<typename T>
void FlowWrappingAggregator<T>::updateFlowCompleteness(const WrapperUnit& unit){
    unsigned int num = windows_fw[unit.window_start_time].completeness;
    unsigned int den = windows_fw[unit.window_start_time].max_completeness;

    if (den < unit.completeness_tag_denominator){
        unsigned int factor = unit.completeness_tag_denominator / den; // we assume this always work for now
        windows_fw[unit.window_start_time].max_completeness *= factor;
        windows_fw[unit.window_start_time].completeness *= factor;
        windows_fw[unit.window_start_time].completeness += unit.completeness_tag_numerator;
    } else if (den == unit.completeness_tag_denominator){
        windows_fw[unit.window_start_time].completeness += unit.completeness_tag_numerator;
    } else {
        throw "[FlowWrappingAggregator](updateCompleteness) Incorrect denominator value for a wrapping unit";
    }
}

/**
 * replaces isWindowComplete from count aggregator
 * */
template<typename T>
bool FlowWrappingAggregator<T>::isWindowComplete(int window_id){
    return windows_fw[window_id].completeness == windows_fw[window_id].max_completeness;
}

/**
 * Reads the wrapper units contained in the header of the message.
 * There can be 1, 2, ... n wrappers depending on how the generator
 * created the message (there can be delays or simply multiple windows
 * in a message).
 * 
 * To know how many wrapper units we have to read, there is an integer
 * in the header that gives us the number of wrapper units.
 * */
template<typename T>
vector<WrapperUnit> FlowWrappingAggregator<T>::extractWrapperUnits(const message_ptr& message){
    vector<WrapperUnit> res;

    int nof_wrapper_units = Serialization::unwrap<int>(message.get());

    for (size_t i = 0; i < nof_wrapper_units; i++){
        WrapperUnit unit = Serialization::unwrap<WrapperUnit>(message.get());
        res.push_back(unit);
    }

    return res;
}
