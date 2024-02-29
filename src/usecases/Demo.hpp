#pragma once
#include "../dataflow/Dataflow.hpp"

using namespace std;

namespace concept_drift_cifar{
class Demo: public Dataflow {

public:

    Vertex *generator, *model, *drift_detector;

    Demo();
	~Demo();


};
};
