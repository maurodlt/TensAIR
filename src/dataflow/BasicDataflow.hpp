#include "Dataflow.hpp"

/**
 * Basic Dataflow to be used on the Python Interface
 * 
 * comm is the MPI variable received from mpi4py
 * operators is a list of operators
 * operatorsLinks are the links of the operators. eg: [[0,1],[1,2]] links operator 0 with operator 1, and operator 1 with operator 2
 */ 

class BasicDataflow : public Dataflow {

    public:
        BasicDataflow(MPI_Comm comm, std::vector<Vertex*> &operators, std::vector<std::tuple<int,int>> &operatorsLinks);
        ~BasicDataflow();
        std::vector<Vertex*> operators; ///list of operators in the dataflow
        std::vector<int> operatorsLinks[2]; ///list of links betwoeen operators in the dataflow
};

///Link and init operators
BasicDataflow::BasicDataflow(MPI_Comm comm, std::vector<Vertex*> &operators, std::vector<std::tuple<int,int>> &operatorsLinks) : Dataflow(comm) {
    int operatorsLinks_size = operatorsLinks.size();
    
    //iterate over links
    for(int i = 0; i <  operatorsLinks_size; i+=1){
        addLink(operators[get<0>(operatorsLinks[i])], operators[get<1>(operatorsLinks[i])]); //add link
    }

    //initialize operators
    for(int i = 0; i < operators.size(); i++){
        operators[i]->initialize();
    }
}

///Default destructor
BasicDataflow::~BasicDataflow() {
    int operators_size = operators.size();
    for(int i = 0; i < operators_size; i++){
         delete operators[i];
    }
    
}
