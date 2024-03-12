#pragma once
#include "../dataflow/Dataflow.hpp"

using namespace std;

namespace vgg16{
class VGG16: public Dataflow {

public:

    Vertex *model;

    VGG16();
	~VGG16();


};
};
