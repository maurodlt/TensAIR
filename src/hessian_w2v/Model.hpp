//#pragma once
//#include "../dataflow/BasicVertex.hpp"
//#include "../dataflow/TensAIR.hpp"
//#include "../hessian_w2v/EventGenerator.hpp"
//#include <map>
//
//namespace hessian_w2v {
//
//struct Tensors_str {
//    int* targets;
//    int* contexts;
//    float* diagonals;
//    float* target_grads;
//    float* context_grads;
//    float* diagonals_grads;
//    int nTargets = 0;
//    int nContexts = 0;
//};
//
//class Model: public TensAIR{
//
//public:
//    
//    Model(const int tag, const int rank, const int worldSize, int mini_batch_size, int batch_window_size, int epochs, float sync_factor, int gpus_per_node, const char* saved_model_dir, const char* tags);
//    //void streamProcess(int channel);
//    
//protected:
//    message_ptr fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list);
//    void fetchUpdateMessages(int channel, list<message_ptr>& lis);
//    vector<output_data> processGradientCalc(message_ptr message);
//    void processGradientApplication(message_ptr message);
//    void predict(Mini_Batch_old ubatch);
//    vector<TF_Tensor*> gradient_calc(Mini_Batch_old ubatch);
//    void apply_gradient(vector<TF_Tensor*> grads);
//    void train_step(Mini_Batch_old ubatch);
//    int number_of_targets(const message_ptr& message);
//    unsigned long int readModelVersion(const message_ptr& message);
//    float readLoss(const message_ptr& message);
//    float readAcc(const message_ptr& message);
//    vector<TF_Tensor*> recreateTensors(Tensors_str targets, float loss, float acc);
//    Mini_Batch_old read_MiniBatch(message_ptr message);
//    pair<map<int,vector<float>>,map<int,vector<float>>> sumGradients(vector<TF_Tensor*> gradients);
//    message_ptr constructMessage_Gradients(vector<TF_Tensor*> out_tensors);
//    message_ptr constructMessage_SumGradients(vector<TF_Tensor*> out_tensors);
//    void progress_bar(int epoch, int new_model_version);
//    void save(string filename);
//    pair<float, float> evaluate(Mini_Batch_old ubatch);
//    pair<float, float> evaluation(vector<Mini_Batch_old> ubatches);
//    vector<Mini_Batch_old> createEvaluationBatches(const char* file, vector<Mini_Batch_old> evaluation_batches);
//    pair<map<int,vector<float>>,map<int,vector<float>>> processGradientWindow(message_ptr message, pair<map<int,vector<float>>,map<int,vector<float>>> gradients, float* mean_loss, float* mean_acc);
//    pair<map<int,vector<float>>,map<int,vector<float>>> sumGradientsWindow(pair<map<int,vector<float>>,map<int,vector<float>>> gradients, pair<map<int,vector<float>>,map<int,vector<float>>> gradientsWindow);
//    vector<output_data> constructMessage_Window(pair<map<int,vector<float>>,map<int,vector<float>>> gradients, float* mean_loss, float* mean_acc);
//
//    int mini_batch_size;
//    int batch_window_size;
//    int epochs;
//    int epochs_generate;
//    int message_type = 0;
//    unsigned long int model_version = 0;
//    unsigned long int *models_version;
//    int epoch = 0;
//    float progress = -0.05;
//    long long int msg_count = 0;
//    float avg_loss = 0;
//    float avg_acc = 0;
//    
//    int embedding_size = EMBEDDING_SIZE;
//    Mini_Batch_old ubatch;
//    vector<Mini_Batch_old> evaluation_batches;
//    
//    list<message_ptr> model_update_list;
//    pthread_mutex_t update_list_mutex = PTHREAD_MUTEX_INITIALIZER;
//    pthread_cond_t update_list_cond = PTHREAD_COND_INITIALIZER;
//    pthread_mutex_t empty_list_mutex = PTHREAD_MUTEX_INITIALIZER;
//    pthread_cond_t empty_list_cond = PTHREAD_COND_INITIALIZER;
//    int model_update_rank = worldSize;
//    
//    
//    //predict
//    const char* predict_input_target = "predict_target"; //(-1,1) int
//    const char* predict_input_context = "predict_context"; //(-1,5,1) int
//    const char* predict_output_similarity = "StatefulPartitionedCall_3"; //(-1,5) float
//
//    //gradient_calc
//    const char* gradient_input_target = "gradient_calc_target"; //(-1,1) int
//    const char* gradient_input_label = "gradient_calc_label"; //(-1,5) int
//    const char* gradient_input_context = "gradient_calc_context"; //(-1,5,1) int
//    const char* gradient_output = "StatefulPartitionedCall_2"; // float
//    
//    //apply_gradient
//    const char* apply_input_loss = "apply_gradient_loss"; //(1) float
//    const char* apply_input_target = "apply_gradient_target"; //(-1) int
//    const char* apply_input_target_gradient = "apply_gradient_target_gradients"; //(-1, emb_dimension) float
//    const char* apply_input_context = "apply_gradient_context"; //(-1) int
//    const char* apply_input_context_gradient = "apply_gradient_context_gradients"; //(-1, emb_dimension) float
//    const char* apply_input_diagonal = "apply_gradient_diagonal0"; //(vocab_size,embedding_dim) float
//    const char* apply_input_diagonal_gradient = "apply_gradient_diagonal1"; //(vocab_size,embedding_dim) float
//    const char* apply_error = "StatefulPartitionedCall"; //(-1, emb_dimension) float
//    
//    //train_step
//    const char* train_input_target = "train_step_target"; //(-1,1) int
//    const char* train_input_label = "train_step_label"; //(-1,5) int
//    const char* train_input_context = "train_step_context"; //(-1,5,1) int
//    const char* train_loss = "StatefulPartitionedCall_5"; // float
//    const char* train_acc = "StatefulPartitionedCall_5"; // float
//    
//    //save
//    const char* save_file = "save_file";
//    const char* output_model = "StatefulPartitionedCall_4";
//    
//    //evaluate
//    const char* evaluate_target_eval = "evaluate_target_eval"; //(-1,1) int
//    const char* evaluate_label_eval = "evaluate_label_eval"; //(-1,5) int
//    const char* evaluate_context_eval = "evaluate_context_eval"; //(-1,5,1) int
//    const char* eval_output = "StatefulPartitionedCall_1"; // float
//    const char* evaluation_file = "../data/W2V/word_embedding_shakespeare_evaluate.txt";
//    //const char* evaluation_file = "../data/W2V/wikipedia1-evaluate.txt"; 
//    
//};
//};
