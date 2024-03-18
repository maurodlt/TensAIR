#pragma once
#include "../dataflow/Dataflow.hpp"

using namespace std;

namespace resnet50{
class RESNET50: public Dataflow {

public:

    Vertex *model;

    RESNET50(int mini_batch_size=128, int epochs=300, int gpus_per_node=0, float loss_objective=0.05);
	~RESNET50();


};
};
