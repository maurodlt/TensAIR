#pragma once
#include "../dataflow/Dataflow.hpp"

using namespace std;

namespace vgg16{
class VGG16: public Dataflow {

public:

    Vertex *model;

    VGG16(int mini_batch_size=128, int epochs=300, int gpus_per_node=0, float loss_objective=0.05);
	~VGG16();


};
};
