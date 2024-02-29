#pragma once
#include "Vertex.hpp"
#include <vector>
#include <memory> // make_unique
#include <sstream>
//#include <pybind11/pybind11.h>

// these unique_ptr will make sure the ownership of the 
// message objects is correctly transferred
using std::move;
using std::make_unique;
using std::pair;
using std::cout;
using message_ptr = unique_ptr<Message>;
using destination = vector<int>;
using output_data = pair<message_ptr, destination>;
//namespace py = pybind11;

/**
 * BasicVertex adds the functionnalities to easily define 
 * and use an operator in a dataflow by defining necessary
 * methods. These methods can be extended or re-defined by
 * subclassing this class, allowing some versatility for 
 * the users.
 * 
 * Template type Event is the type of the input events of 
 * this vertex. In case there is no need to process input 
 * events - such as generator implementation - then it is
 * enough to just leave the default template parameter.
 * 
 * Note when manipulating the message pointers (unique_ptr):
 * - don't forget to use std::move instead of copy when
 *  using =, push_back, a.s.o.
 * - don't use const references to message_ptr when using
 *  std::move
 * */
template<typename Event = char>
class BasicVertex : public Vertex {

    public:
        BasicVertex(const int tag, const int rank, const int worldSize, int windowSize = 1000000, MPI_Comm comm = MPI_COMM_WORLD);
        void streamProcess(int channel);

    protected:
        virtual vector<output_data> processEvent(const Event& event);
        virtual vector<output_data> processMessage(message_ptr message);
        virtual message_ptr fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list);
        virtual void send(vector<output_data> messages);
        void increaseHeaderSize(unsigned int increment);
        void setBaseline(char c);
        unsigned int getHeaderSize();
        message_ptr createMessage(size_t size);
        int readMessageID(const message_ptr& message);
        char readBaseline(const message_ptr& message);
        char getBaseline();
        virtual vector<Event> readEvents(const message_ptr& message);
        vector<message_ptr> fetchMessages(int channel);

        pthread_mutex_t cout_mtx;
        destination target_all_ranks;
        destination target_same_rank;
        destination target_other_ranks;

    private:
        unsigned int header_size = sizeof(int) + sizeof(char); // message id then aggregation protocol (baseline)
        char baseline = 'd'; // d default, c count, w flow-wrapping, s sort
};

#include <numeric>

/**
 * Copy constructor :
 * Creates a BasicVertex instance based on Vertex (rank, worldSize, next, previous, inMessages, outMessages, ...)
 * Initializes the messages' header size and the targets of this operator's output
 * */
template<typename Event>
BasicVertex<Event>::BasicVertex(const int tag, const int rank, const int worldSize, int windowSize, MPI_Comm comm) :
Vertex(tag, rank, worldSize, windowSize, comm),
target_other_ranks(vector<int>()),
target_same_rank(vector<int>({rank})),
target_all_ranks(vector<int>(worldSize)) {
    // fill targetRanks with all possible ranks {0, 1, ... worldSize - 1}
    std::iota(target_all_ranks.begin(), target_all_ranks.end(), 0);

    // initialize mutexes
    pthread_mutex_init(&cout_mtx, NULL);
}

/**
 * Main loop :
 * This is the method that is automatically called to start the dataflow.
 * It keeps in memory the messages received from the input threads and processes
 * these messages one by one.
 * 
 * Note that you can see the different threads working in this method by
 * printing pthread_self() or the channel parameter.
 * 
 * The number of threads running this loop depends on the number of previous
 * operators, the number of dataflow instances (worldSize) and the routing of 
 * the ouput of previous operators.
 * 
 * It is also possible to force the operator to run on one node only by wrapping
 * the streamProcess method in a if(rank == something) condition. This is useful
 * when you want to aggregate the results of all nodes.
 * 
 * Do not forget to add your use case files to the CMake configuration, to create
 * a new use case in the usecase folder and to make your use case useable by
 * modifying the main.cpp as well.
 * */
template<typename Event>
void BasicVertex<Event>::streamProcess(int channel){

    // stores incoming messages
    list<message_ptr> pthread_waiting_list;

    while (ALIVE) {
        message_ptr message = fetchNextMessage(channel, pthread_waiting_list);
        vector<output_data> out = processMessage(std::move(message));
        send(std::move(out));
    }
    
}

/**
 * Fetches the next incoming message. Reads the messages in the same order as
 * they were received. It is possible to modify this behavior by overriding or
 * decorating this method.
 * */
template<typename Event>
message_ptr BasicVertex<Event>::fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list){

    // if there is no message in the waiting list
    // then wait for new messages to arrive
    if (pthread_waiting_list.empty()){
        vector<message_ptr> fetched = std::move(fetchMessages(channel));

        for (size_t i = 0; i < fetched.size(); i++){
            pthread_waiting_list.push_back(std::move(fetched[i]));
        }
    }

    // now we are certain there are messages in the waiting list
    // we can ouput the first one
    message_ptr message = std::move(pthread_waiting_list.front()); // move
    pthread_waiting_list.pop_front();

    return message;
}

/**
 * Fetches new messages from the input buffer.
 * */
template<typename Event>
vector<message_ptr> BasicVertex<Event>::fetchMessages(int channel){
    vector<message_ptr> res; res.reserve(PER_SEC_MSG_COUNT);

    pthread_mutex_lock(&listenerMutexes[channel]);

    // wait until new messages arrive (unlocks the listener mutex
    // until signal is received from listenerConVars[channel])
    while(inMessages[channel].empty()){
        pthread_cond_wait(&listenerCondVars[channel], &listenerMutexes[channel]);
    }

    // when new messages arrive, add them all to the output vector
    while(!inMessages[channel].empty()){
        message_ptr inMessage(inMessages[channel].front());
        res.push_back(std::move(inMessage));
        inMessages[channel].pop_front();
    }

    pthread_mutex_unlock(&listenerMutexes[channel]);

    return std::move(res);
}

/**
 * Reads events one by one and calls a user-defined method to process each event.
 * This effectively moves the user focus from managing messages to processing events.
 * */
template<typename Event>
vector<output_data> BasicVertex<Event>::processMessage(message_ptr message){
    vector<output_data> out;

    int message_id = readMessageID(message);
    char input_baseline = readBaseline(message);

    if(input_baseline != baseline){
        stringstream ss;
        ss << "[BasicVertex](processMessage) Unrecognized aggregation protocol.\n"
        << "protocol sign : " << input_baseline << '\n'
        << "current baseline : " << baseline << endl;
        throw ss.str();
    }

    vector<Event> events = readEvents(message);
    for (const Event& event : events){
        vector<output_data> processed = std::move(processEvent(event));

        for (size_t i = 0; i < processed.size(); i++){
            message_ptr message(std::move(processed[i].first)); // move

            // add message header
            Serialization::wrap<char>(baseline, message.get());
            Serialization::wrap<int>(message_id, message.get());

            destination dest = processed[i].second;
            output_data od = make_pair(std::move(message), dest);
            out.push_back(std::move(od));
        }
    }

    return std::move(out);
}

template<typename Event>
int BasicVertex<Event>::readMessageID(const message_ptr& message){
    // message header = msgid (int) + baseline (char)
    return Serialization::read_back<int>(message.get());
}

template<typename Event>
char BasicVertex<Event>::readBaseline(const message_ptr& message){
    // message header = msgid (int) + baseline (char)
    return Serialization::read_back<char>(message.get(), sizeof(int)); 
}

/**
 * Skips the header of the message buffer and reads all the events it contains.
 * */
template<typename Event>
vector<Event> BasicVertex<Event>::readEvents(const message_ptr& message){
    const size_t nof_events = (message->size - header_size) / sizeof(Event);

    vector<Event> events; events.reserve(nof_events);
    for (size_t i = 0; i < nof_events; i++){
        Event e = Serialization::read_front<Event>(message.get(),  (unsigned int)i * sizeof(Event));
        events.push_back(e); // copy for now
    }

    return events;
}

/**
 * Virtual method the user must define in order to create his own operator.
 * 
 * For example, you can define a data structure in your header file and update
 * it here. Be careful, such data structures can be accessed by all the threads
 * running this method. Use mutexes or make sure the threads never modify the
 * same memory space at the same time.
 * */
template<typename Event>
vector<output_data> BasicVertex<Event>::processEvent(const Event& event){

    // use this mutex every time you use the standard text output
    // as this helps having readable text instead of a mix of different
    // outputs generated by different threads at the same time.
    //
    // you can also use your own mutex that you defined yourself
    // but don't forget to initialize it in the constructor.
    pthread_mutex_lock(&cout_mtx);
    std::stringstream ss; ss << pthread_self();
    std::string s = ss.str();
    cerr << "[BasicVertex](processEvent) basic virtual function has been called by thread " + s + ". Please define your own implementations.\n";
    pthread_mutex_unlock(&cout_mtx);

    vector<output_data> res; res.reserve(1);
    message_ptr out = createMessage(sizeof(char)); // you can also send an empty vector
    destination dest = target_same_rank; // you may want to use target_all_ranks or target_other_ranks and set the destination value somehere else 
    res.push_back(make_pair(std::move(out), dest));

    return std::move(res);
}

/**
 * Initializes a new message with enough space for the header data
 * 
 * Users shoudn't use the natural Message constructor as they would
 * have to take into account the size of the header data as well as 
 * the message's body data every time they wish to create a new
 * message.
 * */
template<typename Event>
message_ptr BasicVertex<Event>::createMessage(size_t size){
    return unique_ptr<Message>(new Message((int)(header_size + size)));
}

/**
 * Sends multiple messages to the next operators in the dataflow.
 * It is possible to modify which ranks will receive the messages
 * by setting up correctly the target_ranks vector.
 * 
 * You can't modify which of the following operators will be delivered
 * with a message, because this is something you have to set up in the
 * use case definition file (e.g. NQ5.hpp and NQ5.cpp)
 * */
template<typename Event>
void BasicVertex<Event>::send(vector<output_data> messages){

    for (output_data& data : messages){

        if (data.second.size() == 0){
            throw "[BasicVertex](send) Message has no destination.";
        } else if (data.second.size() > worldSize) {
            throw "[BasicVertex](send) Message destination rank does not exist.";
        } else {
            for(int targetRank : data.second){
                for(size_t targetOperator = 0; targetOperator < next.size(); targetOperator++){
                    size_t target = targetOperator * worldSize + targetRank;

                    Message* cpy = Serialization::copy(data.first.get());

                    pthread_mutex_lock(&senderMutexes[target]);
                    outMessages[target].push_back(cpy);
                    pthread_cond_signal(&senderCondVars[target]);
                    pthread_mutex_unlock(&senderMutexes[target]);
                }
            }
        }
    }
}

/**
 * Increases the size of the header of the messages created in this vertex.
 * Note that users must use the createMessage method to create new messages
 * or else they have to add header_size to the messages capacity everytime.
 * */
template<typename Event>
void BasicVertex<Event>::increaseHeaderSize(unsigned int increment){
    header_size += increment;
}

/**
 * Returns the current value of the header size. Basic value is sizeof(int)
 * because message header contains an identifier, but it might be increased
 * to contain more data.
 * */
template<typename Event>
unsigned int BasicVertex<Event>::getHeaderSize(){
    return header_size;
}

/**
 * Sets a value for the baseline attribute.
 * 
 * Used when implementing a new aggregator to make sure the user takes 
 * into a account the modifications brought by his new implementation in
 * all the vertices that need it.
 * 
 * e.g. when implementing a new flow-wrapping aggregator, user must make
 * sure that the generator is sending messages with wrapping units and that
 * filter vertices don't modify them, while other vertices might have to.
 * 
 * Here the vertices automatically check if the baseline has been changed on
 * all the vertices of the dataflow. Ideally we should only check once but for
 * now we do it with every message.
 * */
template<typename Event>
void BasicVertex<Event>::setBaseline(char c){
    baseline = c;
}

template<typename Event>
char BasicVertex<Event>::getBaseline(){
    return baseline;
}
