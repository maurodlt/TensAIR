//#include "Hessian_W2V.hpp"
//#include "../hessian_w2v/EventGenerator.hpp"
//#include "../hessian_w2v/Model.hpp"
//#include <tensorflow/c/c_api.h>
//
//using namespace hessian_w2v;
//
//
//Hessian_W2V::Hessian_W2V(int mini_batch_size, int batch_window, int msg_sec, int epochs, float sync_factor, int gpus_per_node) : Dataflow() {
//    generator = new hessian_w2v::EventGenerator(1, rank, worldSize, mini_batch_size, msg_sec, epochs);
//
//    const char* saved_model_dir = "../data/W2V/w2v_shakespeare_hessian.tf";
//    const char* tags = "serve";
//
//    model = new hessian_w2v::Model(2, rank, worldSize, mini_batch_size, batch_window, epochs, sync_factor, gpus_per_node, saved_model_dir, tags);
//
//    addLink(generator, model);
//    addLink(model, model);
//
//    generator->initialize();
//
//    model->initialize();
//
//}
//
//Hessian_W2V::~Hessian_W2V() {
//
//	delete generator;
//	delete model;
//
//}
