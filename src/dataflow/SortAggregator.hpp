#include "CountAggregator.hpp"


/**
 * One of the three baselines for aggregation.
 * Reads messages in order (based on message id).
 * */
template<typename T>
class SortAggregator : public CountAggregator<T> {

    public:
        SortAggregator(int tag, int rank, int worldSize);

    protected:
        message_ptr fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list);
        void updateInputMessages(const int channel, list<message_ptr>& pthread_waiting_list);
        bool isExpected(const message_ptr& message);
        void sortMessages(list<message_ptr>& pthread_waiting_list);

        pthread_mutex_t mtx_sort;

        int previous_wid = -1;
        int expected_msgid = 0;
        int nof_readings = 0;
};

/**
 * Basic constructor extending the count baseline
 * */
template<typename T>
SortAggregator<T>::SortAggregator(int tag, int rank, int worldSize) :
CountAggregator<T>(tag, rank, worldSize){
    pthread_mutex_init(&mtx_sort, NULL);
}

/**
 * The fetching here is different because we want to keep looking
 * for new messages as long as they are not in order, and not simply
 * wait for the first message to arrive.
 * */
template<typename T>
message_ptr SortAggregator<T>::fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list){
    
    while(pthread_waiting_list.empty() || !isExpected(pthread_waiting_list.front())){
        updateInputMessages(channel, pthread_waiting_list);
    }

    message_ptr message = move(pthread_waiting_list.front()); // move expected message
    pthread_waiting_list.pop_front(); // pop empty ptr

    return message;
}

/**
 * This part of the code looks for new messages once
 * and sorts the messages received depending on their 
 * message id value.
 * */
template<typename T>
void SortAggregator<T>::updateInputMessages(int channel, list<message_ptr>& pthread_waiting_list){
    // wait for new messages to arrive
    vector<message_ptr> fetched = BasicVertex<T>::fetchMessages(channel);

    for (size_t i = 0; i < fetched.size(); i++){
        pthread_waiting_list.push_back(move(fetched[i]));
    }

    // now we are certain there are messages in the
    // waiting list, we can sort them.
    sortMessages(pthread_waiting_list);
}

/**
 * Sorts input messages by message id
 * */
template<typename T>
void SortAggregator<T>::sortMessages(list<message_ptr>& pthread_waiting_list){
    pthread_waiting_list.sort(
		[](const message_ptr & a, const message_ptr & b){
		int a_msgid = Serialization::read_back<int>(a.get());
		int b_msgid = Serialization::read_back<int>(b.get());
		
		return (a_msgid < b_msgid);
	});
}

/**
 * Returns true if the message is the one that is expected.
 * Based on message id.
 * 
 * Note : This implementation may vary depending on the 
 * dataflow implementation. For now, we assume the aggregator
 * is supposed to receive all message ids once per previous
 * operator. 
 * (In other implementations, we could have the aggregator 
 * taking care of only half of the message ids, or maybe 
 * each thread will receive different ids so the implementation
 * might change a lot).
 * */
template<typename T>
bool SortAggregator<T>::isExpected(const message_ptr& message){
    int message_id = Serialization::read_back<int>(message.get());

    if (expected_msgid == message_id){

        pthread_mutex_lock(&mtx_sort);
        nof_readings++;

        if (nof_readings == BasicVertex<T>::previous.size()){
            // increment expected message id
            // and reset nof readings
            expected_msgid++;
            nof_readings = 0;
        }

        pthread_mutex_unlock(&mtx_sort);
        return true;
    }
    else {
        return false;
    }
}