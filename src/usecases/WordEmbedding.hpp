#pragma once
#include "../dataflow/Dataflow.hpp"

using namespace std;

namespace word_embedding{
class WordEmbedding: public Dataflow {

public:

    Vertex *generator, *model;

    //WordEmbedding(MPI_Comm comm, int mini_batch_size=512, int batch_window=1, int msg_sec=200, int epochs=10, float sync_factor=0, int gpus_per_node=0);
    //WordEmbedding(int mini_batch_size=512, int batch_window=1, int msg_sec=200, int epochs=10, float sync_factor=0, int gpus_per_node=0);
    WordEmbedding();
	~WordEmbedding();

};
};
