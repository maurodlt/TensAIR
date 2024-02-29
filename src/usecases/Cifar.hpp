#pragma once
#include "../dataflow/Dataflow.hpp"

using namespace std;

namespace cifar{
class Cifar: public Dataflow {

public:

    Vertex *generator, *model, *drift_detector;

    Cifar();
	~Cifar();


};
};
