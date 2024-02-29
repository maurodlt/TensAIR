//#include "Model.hpp"
//#include "../usecases/Hessian_W2V.hpp"
//#include <string>
//#include <typeinfo>
//#include <string.h>
//#include <fstream>
//#include <ctime>
//#include <unistd.h> // usleep
//#include <mpi4py/mpi4py.h>
//#include <sys/resource.h>
//#include <thread> // debug
//#include <tensorflow/c/c_api.h>
//#include <algorithm>
//#include <math.h>
//
//using namespace hessian_w2v;
//
///**
// * Simple constructor.
// * */
//Model::Model(const int tag, const int rank, const int worldSize, int mini_batch_size, int batch_window_size, int epochs, float sync_factor, int gpus_per_node, const char* saved_model_dir, const char* tags) :
//TensAIR(tag, rank,  worldSize, ((((((TRAIN_EXAMPLES+1)*EMBEDDING_SIZE) + 2 + (EMBEDDING_SIZE*300*2)) * mini_batch_size ) + 10 ) * 5), sync_factor, gpus_per_node, saved_model_dir, tags) {
//    //((((((TRAIN_EXAMPLES+1)*EMBEDDING_SIZE) + 2) * mini_batch_size ) + 8 ) * 5) is approximately the size of Tensors_str serialized
//    this->mini_batch_size = mini_batch_size;
//    this->batch_window_size = batch_window_size;
//    this->epochs = epochs;
//    this->epochs_generate = epochs * 2;
//    
//    setBaseline('t');
//    increaseHeaderSize(sizeof(unsigned long int));
//    increaseHeaderSize(sizeof(float)); //loss
//    increaseHeaderSize(sizeof(float)); //acc
//    this->models_version = (unsigned long int*) malloc(sizeof(unsigned long int) * worldSize);
//    for(int i = 0; i < worldSize; i++){
//        this->models_version[i] = (unsigned long int) 0;
//    }
//    
//    evaluation_batches = createEvaluationBatches(evaluation_file, evaluation_batches);
//    //evaluation(evaluation_batches);
//}
//
//
///**
// * We replace the normal streamprocess method because we won't
// * send anything after this vertex.
// *
// * Also, we only need one instance of this collector, just after
// * the aggregator.
// * */
///*void Model::streamProcess(int channel) {
//        //There are worldSize * 2 channels. The first half is from Event Generator (to calculate the gradient) and the second half from Model (to apply the gradient)
//        if(channel < worldSize){ //messages receved from Event Generator. (calculate the gradient)
//            list<message_ptr> pthread_waiting_lists;
//            while(ALIVE){
//                //wait until update list is empty and model is up-to-date
//                while (!model_update_list.empty()) {
//                    pthread_cond_wait(&empty_list_cond, &empty_list_mutex);
//		}
//		//if (pthread_waiting_lists.size()%100 < 10){
//		//   cout << "\n" << pthread_waiting_lists.size() << "\n";
//		//}
//                //calculate gradient
//                message_ptr message = BasicVertex::fetchNextMessage(channel, pthread_waiting_lists);
//                
//                send(move(processGradientCalc(move(message))));
//                msg_count++;
//*/
//                /*//calculate window of mini batches
///*                pair<map<int,vector<float>>,map<int,vector<float>>> gradients;
//                float mean_loss = 0;
//                for(int i = 0; i < batch_window_size; i++){
//                    message_ptr message = BasicVertex::fetchNextMessage(channel, pthread_waiting_lists);
//                    gradients = processGradientWindow(move(message), gradients, &mean_loss);
//                }
//                send(move(constructMessage_Window(gradients, &mean_loss)));
//                msg_count++;
//                */
//                 //sleep(1);
///*            }
//        }else{ //messages received from Model (apply the gradient)
//            while(ALIVE){
//                //end thread - all gradients applied
//                if(epoch == this->epochs_generate){
//                    return;
//                }
//                
//                //update model_update_list
//                fetchUpdateMessages(channel, model_update_list);
//                
//                //one thread applies the updates
//                if(channel == model_update_rank){
//                    message_ptr message = fetchNextMessage(channel, model_update_list);
//                    processGradientApplication(move(message));
//                }
//                
//            }
//        }
//    
//}*/
//
//
//
///**
// * aways searchs for new messages.
// * */
//message_ptr Model::fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list){
//    pthread_mutex_lock(&update_list_mutex);
//
//    message_ptr message = move(pthread_waiting_list.front()); // move
//    pthread_waiting_list.pop_front();
//    
//    pthread_mutex_unlock(&update_list_mutex);
//
//    return message;
//}
//
///**
// * Fetches new messages from the input buffer.
// *     Updates in the model are inserted at the beggining of the list. Then, we train steps are inserted at the end of it.
// * */
//void Model::fetchUpdateMessages(int channel, list<message_ptr>& pthread_waiting_list){
//    pthread_mutex_lock(&listenerMutexes[channel]);
//    
//    //wait until new messages arrive in this or in other channels
//    if(channel == model_update_rank){
//        while(inMessages[channel].empty() && pthread_waiting_list.empty()){
//            pthread_cond_signal(&empty_list_cond); //signal that the model has no updates to perform yet.
//            pthread_cond_wait(&listenerCondVars[channel], &listenerMutexes[channel]);
//        }
//    }else{// wait for messages in this channel
//        while(inMessages[channel].empty()){
//            pthread_cond_signal(&empty_list_cond); //signal that the model has no updates to perform yet.
//            pthread_cond_wait(&listenerCondVars[channel], &listenerMutexes[channel]);
//        }
//    }
//    
//    
//    // when new messages arrive, add them to the list
//    if(!inMessages[channel].empty()){
//        
//        pthread_mutex_lock(&update_list_mutex);
//        while(!inMessages[channel].empty()){
//            message_ptr inMessage(inMessages[channel].front());
//            pthread_waiting_list.push_back(move(inMessage));
//            inMessages[channel].pop_front();
//        }
//        pthread_mutex_unlock(&update_list_mutex);
//        pthread_cond_signal(&listenerCondVars[model_update_rank]); //signal to model_update_rank channel that new messages arrived
//    }
//    pthread_mutex_unlock(&listenerMutexes[channel]);
//    return;
//}
//
//// Iterate over evaluation batches
//pair<float, float> Model::evaluation(vector<Mini_Batch_old> ubatches){
//    isEvaluating = true; //signal to stop calculation of new gradients;
//
//    float acc = 0.0;
//    float loss = 0.0;
//    pair<float,float> eval_output;
//
//    for(int i = 0; i < ubatches.size(); i++){
//        eval_output = evaluate(ubatches[i]);
//        acc += eval_output.first;
//        loss += eval_output.second;
//    }
//    acc = acc / ubatches.size();
//    loss = loss / ubatches.size();
//
//    cout <<  "Rank:" << rank <<" , Movel_version:" << this->models_version[rank] << " - Eval Acc: " << acc << " - Eval Loss: " << loss << "\n";
//
//    isEvaluating = false;
//    pthread_cond_signal(&empty_list_cond); //signal to resume calculation of new gradients;
//
//    return make_pair(acc, loss);
//
//}
//
//
///**
// * Create Evaluation Batches 
// * */
//vector<Mini_Batch_old> Model::createEvaluationBatches(const char* file, vector<Mini_Batch_old> evaluation_batches){
//    bool error = false;
//    
//    //count number of examples in the batch
//    
//    ifstream infile(evaluation_file);
//    int nExamples = count(std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>(), '\n');
//    int nEvalBatches = (int)floor(nExamples/mini_batch_size);
//    infile.close();
//    evaluation_batches.reserve(nEvalBatches);
//    
//    //alloc memory
//    for (int i = 0; i < nEvalBatches; i++){
//        Mini_Batch_old eval_batch;
//        eval_batch.target = (int*) malloc(sizeof(int)*mini_batch_size);
//        eval_batch.label = (int*) malloc(sizeof(int)*mini_batch_size*TRAIN_EXAMPLES);
//        eval_batch.context = (int*) malloc(sizeof(int)*mini_batch_size*TRAIN_EXAMPLES);
//        evaluation_batches.push_back(eval_batch);
//    }
//    
//    
//    int position = 0;
//    int batch_number = 0;
//    ifstream infile2(evaluation_file);
//    while(batch_number < nEvalBatches){
//        while(position < mini_batch_size){
//            
//            infile2 >> evaluation_batches[batch_number].target[position];
//            
//            for (int i = 0; i < TRAIN_EXAMPLES; i++){
//                infile2 >> evaluation_batches[batch_number].context[(position*TRAIN_EXAMPLES)+i];
//            }
//            for (int i = 0; i < TRAIN_EXAMPLES; i++){
//                infile2 >> evaluation_batches[batch_number].label[(position*TRAIN_EXAMPLES) + i];
//            }
//            evaluation_batches[batch_number].size=mini_batch_size;
//
//            //Check if the micro_batch was read successfully
//            if(!infile2){
//                cout << "\n\n ------ Error reading evaluation file --------- \n";
//            }
//            
//                position++;
//        }
//        position = 0;
//        batch_number++; 
//    }
//    infile2.close();
//    
//    return evaluation_batches;
//}
//
//
//
///**
// * Evaluate a mini batch
// * */
//pair<float,float> Model::evaluate(Mini_Batch_old ubatch){
//    int batch_size = ubatch.size;
//    //int emb_dim = this->embedding_size;
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS INPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//    ///////////////////////////////////////////////////////////////////////////////////// Target
//    TF_Operation* target_op = TF_GraphOperationByName(graph, evaluate_target_eval);
//    TF_Output target_opout = {target_op};
//    
//    size_t nbytes_target = batch_size * 1 * sizeof(int);
//    const int64_t target_dims[] = {batch_size, 1};
//    int tarSize = sizeof(target_dims)/sizeof(*target_dims);
//    TF_Tensor* targets_tensor = TF_NewTensor(TF_INT32, target_dims, tarSize, ubatch.target, nbytes_target, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(target_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(targets_tensor, 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Context
//    TF_Operation* context_op = TF_GraphOperationByName(graph, evaluate_context_eval);
//    TF_Output context_opout = {context_op};
//    
//    size_t nbytes_context = batch_size * 5 * 1 * sizeof(int);
//    const int64_t context_dims[] = {batch_size, 5, 1};
//    int contSize = sizeof(context_dims)/sizeof(*context_dims);
//    TF_Tensor* context_tensor = TF_NewTensor(TF_INT32, context_dims, contSize, ubatch.context, nbytes_context, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(context_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(context_tensor, 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Label
//    TF_Operation* label_op = TF_GraphOperationByName(graph, evaluate_label_eval);
//    TF_Output label_opout = {label_op};
//    
//    size_t nbytes_label = batch_size * 5 * sizeof(int);
//    const int64_t label_dims[] = {batch_size, 5};
//    int labelSize = sizeof(label_dims)/sizeof(*label_dims);
//    TF_Tensor* label_tensor = TF_NewTensor(TF_INT32, label_dims, labelSize, ubatch.label, nbytes_label, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(label_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(label_tensor, 0) << "\n";
//    
//    
//    int nDifferentInputs = 3;
//    vector<TF_Output> inputs_opout;
//    inputs_opout.push_back(target_opout);
//    inputs_opout.push_back(context_opout);
//    inputs_opout.push_back(label_opout);
//    
//    vector<TF_Tensor*> input_values;
//    input_values.push_back(targets_tensor);
//    input_values.push_back(context_tensor);
//    input_values.push_back(label_tensor);
//    
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS OUTPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// ACC
//    // Create vector to store graph output operations
//     TF_Operation* out_acc_op = TF_GraphOperationByName(graph, eval_output);
//     TF_Output out_acc_opout = {out_acc_op};
//     out_acc_opout.index=0;
//    
//     // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//     int64_t out_acc_dims[] = {1};
//    
//     // Create variables to store the size of the output variables
//     const int num_bytes_out_acc = sizeof(float);
//     int nSize0 = sizeof(out_acc_dims)/sizeof(*out_acc_dims);
//     TF_Tensor* output_acc_value = TF_AllocateTensor(TF_FLOAT, out_acc_dims, nSize0, num_bytes_out_acc);
//     
//     //// As with inputs, check the values for the output operation and output tensor
//     //std::cout << "Target: " << TF_OperationNumOutputs(out_target_op) << "\n";
//     //std::cout << "Target info: " << TF_Dim(output_target_value, 0) << "\n";
//    
//    
//   
//    /////////////////////////////////////////////////////////////////////////////////////////// LOSS
//    // Create vector to store graph output operations
//     TF_Operation* out_loss_op = TF_GraphOperationByName(graph, eval_output);
//     TF_Output out_loss_opout = {out_loss_op};
//     out_loss_opout.index=1;
//    
//     // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//     int64_t out_loss_dims[] = {1};
//    
//     // Create variables to store the size of the output variables
//     const int num_bytes_out_loss = sizeof(float);
//     int nSize1 = sizeof(out_loss_dims)/sizeof(*out_loss_dims);
//     TF_Tensor* output_loss_value = TF_AllocateTensor(TF_FLOAT, out_loss_dims, nSize0, num_bytes_out_loss);
//     
//     //// As with inputs, check the values for the output operation and output tensor
//     //std::cout << "Target: " << TF_OperationNumOutputs(out_target_op) << "\n";
//     //std::cout << "Target info: " << TF_Dim(output_target_value, 0) << "\n";
//    
//    
//    int nDifferentOutputs = 2;
//    vector<TF_Output> outputs_opout;
//    outputs_opout.push_back(out_acc_opout);
//    outputs_opout.push_back(out_loss_opout);
//    
//    
//    vector<TF_Tensor*> outputs_values;
//    outputs_values.push_back(output_acc_value);
//    outputs_values.push_back(output_loss_value);
//    
//   
//    
//    //////////////////////////////////////////////////////////////////////////// RUN ///////////////////////////////////////////////////////////////////////////////////////////////////////////
//    
//    
//    TF_Tensor** output = runSession(inputs_opout, input_values, nDifferentInputs, outputs_opout, outputs_values, nDifferentOutputs);
//    
//    //////////////////////////////////////////////////////////////////////////// FREE MEMORY //////////////////////////////////////////////////////////////////////////////////////////
//    
//    float acc = static_cast<float*>(TF_TensorData(output[0]))[0];
//    float loss = static_cast<float*>(TF_TensorData(output[1]))[0];
//  
//    
//    for(int i = 0; i < outputs_values.size(); i++){
//        TF_DeleteTensor(outputs_values[i]);
//        TF_DeleteTensor(output[i]);
//    }
//    
//    
//    for (int i = 0; i < nDifferentInputs; i++){
//        TF_DeleteTensor(input_values[i]);
//    }
//    
//    free(output);
//    
//    return make_pair(acc, loss);;
//    
//}
//
//
///**
// * Save model checkpoint
// * */
//void Model::save(string filename){
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS INPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//    TF_Operation* file_op = TF_GraphOperationByName(graph, save_file);
//    TF_Output file_opout = {file_op};
//
//    
//    //send a random int as input. Shall be updated to send the save directory in the future
//    size_t nbytes_file = sizeof(int);
//    const int64_t file_dims[] = {1};
//    int fileSize = sizeof(file_dims)/sizeof(*file_dims);
//    TF_Tensor* file_tensor = TF_NewTensor(TF_INT32, file_dims, fileSize, 0, nbytes_file, &NoOpDeallocator, 0);
//    
//    int nDifferentInputs = 1;
//    vector<TF_Output> inputs_opout;
//    inputs_opout.push_back(file_opout);
//    
//    vector<TF_Tensor*> input_values;
//    input_values.push_back(file_tensor);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(file_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(file_tensor, 0) << "\n";
//    
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS OUTPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//   
//   // Create vector to store graph output operations
//    TF_Operation* output_op = TF_GraphOperationByName(graph, output_model);
//    TF_Output output_opout = {output_op};
//   
//    // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//    int64_t out_dims[] = {1};
//   
//    // Create variables to store the size of the output variables
//    const int num_bytes_out = sizeof(int);
//    int nSize = sizeof(out_dims)/sizeof(*out_dims);
//    TF_Tensor* output_value = TF_AllocateTensor(TF_INT32, out_dims, nSize, num_bytes_out);
//    
//    int nDifferentOutputs = 1;
//    vector<TF_Output> outputs_opout;
//    outputs_opout.push_back(output_opout);
//    
//    vector<TF_Tensor*> outputs_values;
//    outputs_values.push_back(output_value);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Output: " << TF_OperationNumOutputs(output_op) << "\n";
//    //std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
//     
//     //////////////////////////////////////////////////////////////////////////// RUN ///////////////////////////////////////////////////////////////////////////////////////////////////////////
//    
//    
//    TF_Tensor** output = runSession(inputs_opout, input_values, nDifferentInputs, outputs_opout, outputs_values, nDifferentOutputs);
//
//    //////////////////////////////////////////////////////////////////////////// FREE MEMORY //////////////////////////////////////////////////////////////////////////////////////////
//    
//    for (int i = 0; i < nDifferentInputs; i++){
//        TF_DeleteTensor(input_values[i]);
//    }
//    
//    for (int i = 0; i < nDifferentOutputs; i++){
//        TF_DeleteTensor(output[i]);
//    }
//    //free(out_vals);
//    free(output);
//    return;
//    
//}
//
//
///**
// * Make a prediction based on the current model
// * */
//void Model::predict(Mini_Batch_old ubatch){
//    int batch_size = ubatch.size;
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS INPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//    ///////////////////////////////////////////////////////////////////////////////////// Target
//    
//    TF_Operation* target_op = TF_GraphOperationByName(graph, predict_input_target);
//    TF_Output target_opout = {target_op};
//    
//    size_t nbytes_target = batch_size * 1 * sizeof(int);
//    const int64_t target_dims[] = {batch_size, 1};
//    int tarSize = sizeof(target_dims)/sizeof(*target_dims);
//    TF_Tensor* targets_tensor = TF_NewTensor(TF_INT32, target_dims, tarSize, ubatch.target, nbytes_target, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(target_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(targets_tensor, 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Context
//    TF_Operation* context_op = TF_GraphOperationByName(graph, predict_input_context);
//    TF_Output context_opout = {context_op};
//
//    size_t nbytes_context = batch_size * 5 * 1 * sizeof(int) ;
//    const int64_t context_dims[] = {batch_size, 5, 1};
//    int contSize = sizeof(context_dims)/sizeof(*context_dims);
//    TF_Tensor* context_tensor = TF_NewTensor(TF_INT32, context_dims, contSize, ubatch.context, nbytes_context, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Context: " << TF_OperationNumOutputs(context_op) << "\n";
//    //std::cout << "Context info: " << TF_Dim(context_tensor, 0) << "\n";
//    
//    
//    int nDifferentInputs = 2;
//    vector<TF_Output> inputs_opout;
//    inputs_opout.push_back(target_opout);
//    inputs_opout.push_back(context_opout);
//    
//    vector<TF_Tensor*> input_values;
//    input_values.push_back(targets_tensor);
//    input_values.push_back(context_tensor);
//    
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS OUTPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//   
//   
//   // Create vector to store graph output operations
//    TF_Operation* output_op = TF_GraphOperationByName(graph, predict_output_similarity);
//    TF_Output output_opout = {output_op};
//   
//    // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//    int64_t out_dims[] = {batch_size, 5};
//   
//    // Create variables to store the size of the output variables
//    const int num_bytes_out = batch_size * 5 * 1 * sizeof(float);
//    int nSize = sizeof(out_dims)/sizeof(*out_dims);
//    TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, nSize, num_bytes_out);
//
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Output: " << TF_OperationNumOutputs(output_op) << "\n";
//    //std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
//    
//    int nDifferentOutputs = 1;
//    vector<TF_Output> outputs_opout;
//    outputs_opout.push_back(output_opout);
//    
//    vector<TF_Tensor*> outputs_values;
//    outputs_values.push_back(output_value);
//   
//    
//    //////////////////////////////////////////////////////////////////////////// RUN ///////////////////////////////////////////////////////////////////////////////////////////////////////////
//    
//    
//    TF_Tensor** output = runSession(inputs_opout, input_values, nDifferentInputs, outputs_opout, outputs_values, nDifferentOutputs);
//
//    float* out_vals = static_cast<float*>(TF_TensorData(output[0]));
//    
//    cout <<  "Rank:" << rank <<" , Movel_version:" << this->models_version[rank] << " - Predict values: " ;
//    for (int i = 0; i < 5; ++i)
//    {
//        std::cout << *out_vals++ << "   ";
//    }
//    std::cout << "\n";
//    //////////////////////////////////////////////////////////////////////////// FREE MEMORY //////////////////////////////////////////////////////////////////////////////////////////
//    
//    for (int i = 0; i < nDifferentInputs; i++){
//        TF_DeleteTensor(input_values[i]);
//    }
//    
//    for (int i = 0; i < nDifferentOutputs; i++){
//        TF_DeleteTensor(outputs_values[i]);
//        TF_DeleteTensor(output[i]);
//    }
//    //free(out_vals);
//    free(output);
//    return;
//}
//
///**
// * Calculate a micro_batch gradient
// * */
//vector<TF_Tensor*> Model::gradient_calc(Mini_Batch_old ubatch){
//   
//    int batch_size = ubatch.size;
//    int emb_dim = this->embedding_size;
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS INPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//    ///////////////////////////////////////////////////////////////////////////////////// Target
//    TF_Operation* target_op = TF_GraphOperationByName(graph, gradient_input_target);
//    TF_Output target_opout = {target_op};
//    
//    size_t nbytes_target = batch_size * 1 * sizeof(int);
//    const int64_t target_dims[] = {batch_size, 1};
//    int tarSize = sizeof(target_dims)/sizeof(*target_dims);
//    TF_Tensor* targets_tensor = TF_NewTensor(TF_INT32, target_dims, tarSize, ubatch.target, nbytes_target, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(target_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(targets_tensor, 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Context
//    TF_Operation* context_op = TF_GraphOperationByName(graph, gradient_input_context);
//    TF_Output context_opout = {context_op};
//    
//    size_t nbytes_context = batch_size * 5 * 1 * sizeof(int);
//    const int64_t context_dims[] = {batch_size, 5, 1};
//    int contSize = sizeof(context_dims)/sizeof(*context_dims);
//    TF_Tensor* context_tensor = TF_NewTensor(TF_INT32, context_dims, contSize, ubatch.context, nbytes_context, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(context_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(context_tensor, 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Label
//    TF_Operation* label_op = TF_GraphOperationByName(graph, gradient_input_label);
//    TF_Output label_opout = {label_op};
//    
//    size_t nbytes_label = batch_size * 5 * sizeof(int);
//    const int64_t label_dims[] = {batch_size, 5};
//    int labelSize = sizeof(label_dims)/sizeof(*label_dims);
//    TF_Tensor* label_tensor = TF_NewTensor(TF_INT32, label_dims, labelSize, ubatch.label, nbytes_label, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(label_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(label_tensor, 0) << "\n";
//    
//    
//    int nDifferentInputs = 3;
//    vector<TF_Output> inputs_opout;
//    inputs_opout.push_back(target_opout);
//    inputs_opout.push_back(context_opout);
//    inputs_opout.push_back(label_opout);
//    
//    vector<TF_Tensor*> input_values;
//    input_values.push_back(targets_tensor);
//    input_values.push_back(context_tensor);
//    input_values.push_back(label_tensor);
//    
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS OUTPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// LOSS
//    // Create vector to store graph output operations
//     TF_Operation* out_loss_op = TF_GraphOperationByName(graph, gradient_output);
//     TF_Output out_loss_opout = {out_loss_op};
//     out_loss_opout.index=0;
//    
//     // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//     int64_t out_loss_dims[] = {1};
//    
//     // Create variables to store the size of the output variables
//     const int num_bytes_out_loss = sizeof(float);
//     int nSize0 = sizeof(out_loss_dims)/sizeof(*out_loss_dims);
//     TF_Tensor* output_loss_value = TF_AllocateTensor(TF_FLOAT, out_loss_dims, nSize0, num_bytes_out_loss);
//     
//     //// As with inputs, check the values for the output operation and output tensor
//     //std::cout << "Target: " << TF_OperationNumOutputs(out_target_op) << "\n";
//     //std::cout << "Target info: " << TF_Dim(output_target_value, 0) << "\n";
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// ACC
//    // Create vector to store graph output operations
//     TF_Operation* out_acc_op = TF_GraphOperationByName(graph, gradient_output);
//     TF_Output out_acc_opout = {out_acc_op};
//     out_acc_opout.index=1;
//    
//     // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//     int64_t out_acc_dims[] = {1};
//    
//     // Create variables to store the size of the output variables
//     const int num_bytes_out_acc = sizeof(float);
//     nSize0 = sizeof(out_acc_dims)/sizeof(*out_acc_dims);
//     TF_Tensor* output_acc_value = TF_AllocateTensor(TF_FLOAT, out_acc_dims, nSize0, num_bytes_out_acc);
//     
//     //// As with inputs, check the values for the output operation and output tensor
//     //std::cout << "Target: " << TF_OperationNumOutputs(out_target_op) << "\n";
//     //std::cout << "Target info: " << TF_Dim(output_target_value, 0) << "\n";
//    
//    
//   
//   /////////////////////////////////////////////////////////////////////////////////////////// TARGET
//   // Create vector to store graph output operations
//    TF_Operation* out_target_op = TF_GraphOperationByName(graph, gradient_output);
//    TF_Output out_target_opout = {out_target_op};
//    out_target_opout.index=2;
//   
//    // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//    int64_t out_target_dims[] = {batch_size};
//   
//    // Create variables to store the size of the output variables
//    const int num_bytes_out_target = batch_size * sizeof(int);
//    int nSize = sizeof(out_target_dims)/sizeof(*out_target_dims);
//    TF_Tensor* output_target_value = TF_AllocateTensor(TF_INT32, out_target_dims, nSize, num_bytes_out_target);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(out_target_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(output_target_value, 0) << "\n";
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// TARGET Gradient
//    // Create vector to store graph output operations
//     TF_Operation* out_target_grad_op = TF_GraphOperationByName(graph, gradient_output);
//     TF_Output out_target_grad_opout = {out_target_grad_op};
//     out_target_grad_opout.index=3;
//    
//     // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//     int64_t out_target_grad_dims[] = {batch_size, emb_dim};
//    
//     // Create variables to store the size of the output variables
//     const int num_bytes_out_target_grad = batch_size * emb_dim * sizeof(float);
//     int nSize2 = sizeof(out_target_grad_dims)/sizeof(*out_target_grad_dims);
//     TF_Tensor* output_target_grad_value = TF_AllocateTensor(TF_FLOAT, out_target_grad_dims, nSize2, num_bytes_out_target_grad);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(out_target_grad_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(output_target_grad_value, 0) << "\n";
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// CONTEXT
//    // Create vector to store graph output operations
//     TF_Operation* out_context_op = TF_GraphOperationByName(graph, gradient_output);
//     TF_Output out_context_opout = {out_context_op};
//     out_context_opout.index=4;
//    
//     // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//     int64_t out_context_dims[] = {5 * batch_size};
//    
//     // Create variables to store the size of the output variables
//     const int num_bytes_out_context = 5 * batch_size * sizeof(int);
//     int nSize3 = sizeof(out_context_dims)/sizeof(*out_context_dims);
//     TF_Tensor* output_context_value = TF_AllocateTensor(TF_INT32, out_context_dims, nSize3, num_bytes_out_context);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(out_context_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(output_context_value, 0) << "\n";
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// CONTEXT Gradient
//    // Create vector to store graph output operations
//     TF_Operation* out_context_grad_op = TF_GraphOperationByName(graph, gradient_output);
//     TF_Output out_context_grad_opout = {out_context_grad_op};
//     out_context_grad_opout.index=5;
//    
//     // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//     int64_t out_context_grad_dims[] = {5 * batch_size, emb_dim};
//    
//     // Create variables to store the size of the output variables
//     const int num_bytes_out_context_grad = 5 * batch_size * emb_dim * sizeof(float);
//     int nSize4 = sizeof(out_context_grad_dims)/sizeof(*out_context_grad_dims);
//     TF_Tensor* output_context_grad_value = TF_AllocateTensor(TF_FLOAT, out_context_grad_dims, nSize4, num_bytes_out_context_grad);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(out_context_grad_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(output_context_grad_value, 0) << "\n";
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// DIAGONAL
//    // Create vector to store graph output operations
//     TF_Operation* out_diagonal_op = TF_GraphOperationByName(graph, gradient_output);
//     TF_Output out_diagonal_opout = {out_diagonal_op};
//     out_diagonal_opout.index=6;
//    
//     // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//     int64_t out_diagonal_dims[] = {50000,300};
//    
//     // Create variables to store the size of the output variables
//     const int num_bytes_out_diagonal = sizeof(float) * 300 * 50000;
//     int nSize5 = sizeof(out_diagonal_dims)/sizeof(*out_diagonal_dims);
//     TF_Tensor* output_diagonal_value = TF_AllocateTensor(TF_FLOAT, out_diagonal_dims, nSize5, num_bytes_out_diagonal);
//     
//     //// As with inputs, check the values for the output operation and output tensor
//     //std::cout << "Target: " << TF_OperationNumOutputs(out_target_op) << "\n";
//     //std::cout << "Target info: " << TF_Dim(output_target_value, 0) << "\n";
//     
//     /////////////////////////////////////////////////////////////////////////////////////////// DIAGONAL Gradient
//     // Create vector to store graph output operations
//      TF_Operation* out_diagonal_grad_op = TF_GraphOperationByName(graph, gradient_output);
//      TF_Output out_diagonal_grad_opout = {out_diagonal_grad_op};
//      out_diagonal_grad_opout.index=7;
//     
//      // Set output dimensions - this should match the dimensionality of the input in the loaded graph
//      int64_t out_diagonal_grad_dims[] = {50000,300};
//     
//      // Create variables to store the size of the output variables
//      const int num_bytes_out_diagonal_grad = sizeof(float) * 300 * 50000;
//      int nSize6 = sizeof(out_diagonal_grad_dims)/sizeof(*out_diagonal_grad_dims);
//      TF_Tensor* output_diagonal_grad_value = TF_AllocateTensor(TF_FLOAT, out_diagonal_grad_dims, nSize6, num_bytes_out_diagonal_grad);
//     
//     //// As with inputs, check the values for the output operation and output tensor
//     //std::cout << "Target: " << TF_OperationNumOutputs(out_target_grad_op) << "\n";
//     //std::cout << "Target info: " << TF_Dim(output_target_grad_value, 0) << "\n";
//    
//    
//    int nDifferentOutputs = 8;
//    vector<TF_Output> outputs_opout;
//    outputs_opout.push_back(out_loss_opout);
//    outputs_opout.push_back(out_acc_opout);
//    outputs_opout.push_back(out_target_opout);
//    outputs_opout.push_back(out_target_grad_opout);
//    outputs_opout.push_back(out_context_opout);
//    outputs_opout.push_back(out_context_grad_opout);
//    outputs_opout.push_back(out_diagonal_opout);
//    outputs_opout.push_back(out_diagonal_grad_opout);
//    
//    vector<TF_Tensor*> outputs_values;
//    outputs_values.push_back(output_loss_value);
//    outputs_values.push_back(output_acc_value);
//    outputs_values.push_back(output_target_value);
//    outputs_values.push_back(output_target_grad_value);
//    outputs_values.push_back(output_context_value);
//    outputs_values.push_back(output_context_grad_value);
//    outputs_values.push_back(output_diagonal_value);
//    outputs_values.push_back(output_diagonal_grad_value);
//   
//    
//    //////////////////////////////////////////////////////////////////////////// RUN ///////////////////////////////////////////////////////////////////////////////////////////////////////////
//    
//    
//    TF_Tensor** output = runSession(inputs_opout, input_values, nDifferentInputs, outputs_opout, outputs_values, nDifferentOutputs);
//    
//    //////////////////////////////////////////////////////////////////////////// FREE MEMORY //////////////////////////////////////////////////////////////////////////////////////////
//    
//    for(int i = 0; i < outputs_values.size(); i++){
//        TF_DeleteTensor(outputs_values[i]);
//        outputs_values[i] = output[i];
//    }
//    
//    
//    for (int i = 0; i < nDifferentInputs; i++){
//        TF_DeleteTensor(input_values[i]);
//    }
//    
//    free(output);
//    
//    return outputs_values;
//    
//}
//
///**
// * Apply gradient
// * */
//void Model::apply_gradient(vector<TF_Tensor*> grads){
//    int nDifferentInputs = 6;
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS INPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Target
//    TF_Operation* target_op = TF_GraphOperationByName(graph, apply_input_target);
//    TF_Output target_opout = {target_op};
//    target_opout.index = 0;
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Output: " << TF_OperationNumOutputs(target_op) << "\n";
//    //std::cout << "Output info: " << TF_Dim(grads[0], 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Target Gradient
//    TF_Operation* target_grad_op = TF_GraphOperationByName(graph, apply_input_target_gradient);
//    TF_Output target_grad_opout = {target_grad_op};
//    target_grad_opout.index = 0;
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Output: " << TF_OperationNumOutputs(target_grad_op) << "\n";
//    //std::cout << "Output info: " << TF_Dim(grads[1], 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Context
//    TF_Operation* context_op = TF_GraphOperationByName(graph, apply_input_context);
//    TF_Output context_opout = {context_op};
//    context_opout.index = 0;
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Output: " << TF_OperationNumOutputs(context_op) << "\n";
//    //std::cout << "Output info: " << TF_Dim(grads[2], 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Context Gradient
//    TF_Operation* context_grad_op = TF_GraphOperationByName(graph, apply_input_context_gradient);
//    TF_Output context_grad_opout = {context_grad_op};
//    context_grad_opout.index = 0;
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Output: " << TF_OperationNumOutputs(context_grad_op) << "\n";
//    //std::cout << "Output info: " << TF_Dim(grads[3], 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Diagonal
//    TF_Operation* diagonal_op = TF_GraphOperationByName(graph, apply_input_diagonal);
//    TF_Output diagonal_opout = {diagonal_op};
//    diagonal_opout.index = 0;
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Diagonal0: " << TF_OperationNumOutputs(diagonal_op) << "\n";
//    //std::cout << "Diagonal0 info: " << TF_Dim(grads[4], 1) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Diagonal Gradient
//    TF_Operation* diagonal_grad_op = TF_GraphOperationByName(graph, apply_input_diagonal_gradient);
//    TF_Output diagonal_grad_opout = {diagonal_grad_op};
//    diagonal_grad_opout.index = 0;
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Diagonal1: " << TF_OperationNumOutputs(diagonal_grad_op) << "\n";
//    //std::cout << "Diagonal1 info: " << TF_Dim(grads[5], 1) << "\n";
//
//    vector<TF_Output> inputs_opout;
//    inputs_opout.push_back(target_opout);
//    inputs_opout.push_back(target_grad_opout);
//    inputs_opout.push_back(context_opout);
//    inputs_opout.push_back(context_grad_opout);
//    inputs_opout.push_back(diagonal_opout);
//    inputs_opout.push_back(diagonal_grad_opout);
//    
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS OUTPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//  
//    //////////////////////////////////////////////////////////////////////////////////////// ACC
//    TF_Operation* acc_op = TF_GraphOperationByName(graph, apply_error);
//    TF_Output acc_opout = {acc_op};
//    acc_opout.index = 0;
//    
//    int64_t acc_dims[] = {1};
//    const int num_bytes_acc = sizeof(float);
//    int nSize_acc = sizeof(acc_dims)/sizeof(*acc_dims);
//    TF_Tensor* acc_value = TF_AllocateTensor(TF_FLOAT, acc_dims, 1, num_bytes_acc);
//
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "ACC: " << TF_OperationNumOutputs(acc_op) << "\n";
//    //std::cout << "ACC info: " << TF_Dim(acc_value, 0) << "\n";
//   
//    ///////////////////////////////////////////////////////////////////////////////////// LOSS
//    TF_Operation* error_op = TF_GraphOperationByName(graph, apply_error);
//    TF_Output error_opout = {error_op};
//    error_opout.index = 1;
//    
//    int64_t error_dims[] = {1};
//    const int num_bytes_error = sizeof(float);
//    int nSize = sizeof(error_dims)/sizeof(*error_dims);
//    TF_Tensor* error_value = TF_AllocateTensor(TF_FLOAT, error_dims, 1, num_bytes_error);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Output: " << TF_OperationNumOutputs(error_op) << "\n";
//    //std::cout << "Output info: " << TF_Dim(error_value, 0) << "\n";
//    
//    
//    int nDifferentOutputs = 2;
//    vector<TF_Output> outputs_opout;
//    outputs_opout.push_back(acc_opout);
//    outputs_opout.push_back(error_opout);
//    vector<TF_Tensor*> outputs_values;
//    outputs_values.push_back(acc_value);
//    outputs_values.push_back(error_value);
//    
//    
//    
//    //////////////////////////////////////////////////////////////////////////// RUN ///////////////////////////////////////////////////////////////////////////////////////////////////////////
//    
//    
//    TF_Tensor** output = runSession(inputs_opout, grads, nDifferentInputs, outputs_opout, outputs_values, nDifferentOutputs);
//    
//    
//    //////////////////////////////////////////////////////////////////////////// FREE MEMORY //////////////////////////////////////////////////////////////////////////////////////////
//    
//    for (int i = 0; i < nDifferentInputs; i++){
//        TF_DeleteTensor(grads[i]);
//    }
//    
//    for (int i = 0; i < nDifferentOutputs; i++){
//        TF_DeleteTensor(outputs_values[i]);
//        TF_DeleteTensor(output[i]);
//    }
//    
//    free(output);
//    
//    return;
//}
//
///**
// * Full train-step. Calculate and apply gradient
// * */
//void Model::train_step(Mini_Batch_old ubatch){
//    int batch_size = ubatch.size;
//    //int emb_dim = this->embedding_size;
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS INPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//    ///////////////////////////////////////////////////////////////////////////////////// Target
//    TF_Operation* target_op = TF_GraphOperationByName(graph, train_input_target);
//    TF_Output target_opout = {target_op};
//    target_opout.index = 0;
//
//    size_t nbytes_target = batch_size * 1 * sizeof(int);
//    const int64_t target_dims[] = {batch_size, 1};
//    int tarSize = sizeof(target_dims)/sizeof(*target_dims);
//    TF_Tensor* targets_tensor = TF_NewTensor(TF_INT32, target_dims, tarSize, ubatch.target, nbytes_target, &NoOpDeallocator, 0);
//    
//    //std::memcpy(TF_TensorData(targets_tensor), ubatch.target, tarSize);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(target_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(targets_tensor, 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Context
//    TF_Operation* context_op = TF_GraphOperationByName(graph, train_input_context);
//    TF_Output context_opout = {context_op};
//    context_opout.index = 0;
//    
//    size_t nbytes_context = batch_size * 5 * 1 * sizeof(int);
//    const int64_t context_dims[] = {batch_size, 5, 1};
//    int contSize = sizeof(context_dims)/sizeof(*context_dims);
//    TF_Tensor* context_tensor = TF_NewTensor(TF_INT32, context_dims, contSize, ubatch.context, nbytes_context, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(context_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(context_tensor, 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// Label
//    TF_Operation* label_op = TF_GraphOperationByName(graph, train_input_label);
//    TF_Output label_opout = {label_op};
//    label_opout.index = 0;
//    
//    size_t nbytes_label = batch_size * 5 * sizeof(int);
//    const int64_t label_dims[] = {batch_size, 5};
//    int labelSize = sizeof(label_dims)/sizeof(*label_dims);
//    TF_Tensor* label_tensor = TF_NewTensor(TF_INT32, label_dims, labelSize, ubatch.label, nbytes_label, &NoOpDeallocator, 0);
//    
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Target: " << TF_OperationNumOutputs(label_op) << "\n";
//    //std::cout << "Target info: " << TF_Dim(label_tensor, 0) << "\n";
//    
//    
//    int nDifferentInputs = 3;
//    vector<TF_Output> inputs_opout;
//    inputs_opout.push_back(target_opout);
//    inputs_opout.push_back(context_opout);
//    inputs_opout.push_back(label_opout);
//    
//    vector<TF_Tensor*> input_values;
//    input_values.push_back(targets_tensor);
//    input_values.push_back(context_tensor);
//    input_values.push_back(label_tensor);
//    
//    
//    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS OUTPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
//   
//    ///////////////////////////////////////////////////////////////////////////////////// LOSS
//    TF_Operation* loss_op = TF_GraphOperationByName(graph, train_loss);
//    TF_Output loss_opout = {loss_op};
//    loss_opout.index = 0;
//    
//    int64_t loss_dims[] = {1};
//    const int num_bytes_loss = sizeof(float);
//    int nSize = sizeof(loss_dims)/sizeof(*loss_dims);
//    TF_Tensor* loss_value = TF_AllocateTensor(TF_FLOAT, loss_dims, 1, num_bytes_loss);
//
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Output: " << TF_OperationNumOutputs(output_op) << "\n";
//    //std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
//    
//    ///////////////////////////////////////////////////////////////////////////////////// ACC
//    TF_Operation* acc_op = TF_GraphOperationByName(graph, train_acc);
//    TF_Output acc_opout = {acc_op};
//    acc_opout.index = 1;
//    
//    int64_t acc_dims[] = {1};
//    const int num_bytes_acc = sizeof(float);
//    int nSize1 = sizeof(acc_dims)/sizeof(*acc_dims);
//    TF_Tensor* acc_value = TF_AllocateTensor(TF_FLOAT, acc_dims, 1, num_bytes_acc);
//
//    //// As with inputs, check the values for the output operation and output tensor
//    //std::cout << "Output: " << TF_OperationNumOutputs(output_op) << "\n";
//    //std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
//    
//    
//    int nDifferentOutputs = 2;
//    vector<TF_Output> outputs_opout;
//    outputs_opout.push_back(loss_opout);
//    outputs_opout.push_back(acc_opout);
//    
//    vector<TF_Tensor*> outputs_values;
//    outputs_values.push_back(loss_value);
//    outputs_values.push_back(acc_value);
//   
//    
//    //////////////////////////////////////////////////////////////////////////// RUN ///////////////////////////////////////////////////////////////////////////////////////////////////////////
//    
//    
//    TF_Tensor** output = runSession(inputs_opout, input_values, nDifferentInputs, outputs_opout, outputs_values, nDifferentOutputs);
//
//    //float* out_vals = static_cast<float*>(TF_TensorData(output[0]));
//    //std::cout << "Loss: " << *out_vals << "     - Model Version:" << this->models_version[rank]   << "     -   Inference Rank: " << rank <<"\n";
//    
//    this->models_version[rank]++;
//
//    
//    //////////////////////////////////////////////////////////////////////////// FREE MEMORY //////////////////////////////////////////////////////////////////////////////////////////
//    
//    for (int i = 0; i < nDifferentInputs; i++){
//        TF_DeleteTensor(input_values[i]);
//    }
//    
//    for (int i = 0; i < nDifferentOutputs; i++){
//        TF_DeleteTensor(outputs_values[i]);
//        TF_DeleteTensor(output[i]);
//        
//    }
//    //free(out_vals);
//    free(output);
//}
//
//float Model::readLoss(const message_ptr& message){
//    // message header = #targets (int) + baseline(char) + model_version (unsigned long int) + acc (float)
//    return Serialization::read_back<float>(message.get(), sizeof(char) + sizeof(int) + sizeof(unsigned long int) + sizeof(float));
//}
//
//float Model::readAcc(const message_ptr& message){
//    // message header = #targets (int) + baseline(char) + model_version (unsigned long int) + acc (float)
//    return Serialization::read_back<float>(message.get(), sizeof(char) + sizeof(int) + sizeof(unsigned long int));
//}
//
//
//unsigned long int Model::readModelVersion(const message_ptr& message){
//    // message header = #targets (int) + baseline(char) + model_version (unsigned long int) + acc (float)
//    return Serialization::read_back<unsigned long int>(message.get(), sizeof(char) + sizeof(int));
//}
//
//int Model::number_of_targets(const message_ptr& message){
//    // message header = #targets (int) + model_version (unsigned long int) + acc (float)
//    return Serialization::read_back<int>(message.get());
//}
//
///**
// * Recreate the gradients in Tensor objects to use them in the apply_gradient method
// * */
//vector<TF_Tensor*> Model::recreateTensors(Tensors_str tensors, float loss, float acc){
//    
//    ///////////////////////////////////////////////////////// RECREATE TARGET TENSOR
//    size_t nbytes_target = tensors.nTargets * sizeof(int);
//    const int64_t target_dims[] = {tensors.nTargets};
//    int tarSize = sizeof(target_dims)/sizeof(*target_dims);
//    TF_Tensor* targets_tensor = TF_NewTensor(TF_INT32, target_dims, tarSize, tensors.targets, nbytes_target, &NoOpDeallocator, 0);
//    //std::cout << "\nOutput info: " << TF_Dim(targets_tensor, 0)  << "  -  " <<  TF_NumDims(targets_tensor) << "  -  " << TF_TensorElementCount(targets_tensor) << "  -  " << TF_TensorType(targets_tensor) <<"\n";
//    
//    ///////////////////////////////////////////////////////// RECREATE TARGET GRADIENT
//    size_t nbytes_target_grad = tensors.nTargets * EMBEDDING_SIZE * sizeof(float);
//    const int64_t target_grad_dims[] = {tensors.nTargets, EMBEDDING_SIZE};
//    int tar_gradSize = sizeof(target_grad_dims)/sizeof(*target_grad_dims);
//    TF_Tensor* targets_gradients_tensor = TF_NewTensor(TF_FLOAT, target_grad_dims, tar_gradSize, tensors.target_grads, nbytes_target_grad, &NoOpDeallocator, 0);
//    //std::cout << "Output info: " << TF_Dim(targets_gradients_tensor, 0) << " , " << TF_Dim(targets_gradients_tensor, 1) << "  -  " <<  TF_NumDims(targets_gradients_tensor) << "  -  " << TF_TensorElementCount(targets_gradients_tensor) << "  -  " << TF_TensorType(targets_gradients_tensor) <<"\n";
//    
//    
//    ///////////////////////////////////////////////////////// RECREATE CONTEXT TENSOR
//    size_t nbytes_context = tensors.nContexts * sizeof(int);
//    const int64_t context_dims[] = {tensors.nContexts};
//    int contSize = sizeof(context_dims)/sizeof(*context_dims);
//    TF_Tensor* contexts_tensor = TF_NewTensor(TF_INT32, context_dims, contSize, tensors.contexts, nbytes_context, &NoOpDeallocator, 0);
//    //std::cout << "Output info: " << TF_Dim(contexts_tensor, 0)  << "  -  " <<  TF_NumDims(contexts_tensor) << "  -  " << TF_TensorElementCount(contexts_tensor) << "  -  " << TF_TensorType(contexts_tensor) <<"\n";
//    
//    
//    ///////////////////////////////////////////////////////// RECREATE CONTEXT GRADIENT
//    size_t nbytes_context_grad = tensors.nContexts * EMBEDDING_SIZE * sizeof(float);
//    const int64_t context_grad_dims[] = {tensors.nContexts, EMBEDDING_SIZE};
//    int cont_gradSize = sizeof(context_grad_dims)/sizeof(*context_grad_dims);
//    TF_Tensor* contexts_gradients_tensor = TF_NewTensor(TF_FLOAT, context_grad_dims, cont_gradSize, tensors.context_grads, nbytes_context_grad, &NoOpDeallocator, 0);
//    //std::cout << "Output info: " << TF_Dim(contexts_gradients_tensor, 0) << " , " << TF_Dim(contexts_gradients_tensor, 1) << "  -  " <<  TF_NumDims(contexts_gradients_tensor) << "  -  " << TF_TensorElementCount(contexts_gradients_tensor) << "  -  " << TF_TensorType(contexts_gradients_tensor) <<"\n";
//    
//    ///////////////////////////////////////////////////////// RECREATE DIAGONAL0 TENSOR
//    size_t nbytes_diagonal0 = sizeof(float) * EMBEDDING_SIZE * 50000;
//    const int64_t diagonal0_dims[] = {50000,EMBEDDING_SIZE};
//    int diag0Size = sizeof(diagonal0_dims)/sizeof(*diagonal0_dims);
//    TF_Tensor* diagonal0_tensor = TF_NewTensor(TF_FLOAT, diagonal0_dims, diag0Size, tensors.diagonals, nbytes_diagonal0, &NoOpDeallocator, 0);
//    //std::cout << "Output info: " << TF_Dim(diagonal0_tensor, 0)  << "  -  " <<  TF_NumDims(diagonal0_tensor) << "  -  " << TF_TensorElementCount(diagonal0_tensor) << "  -  " << TF_TensorType(diagonal0_tensor) <<"\n";
//    
//    
//    ///////////////////////////////////////////////////////// RECREATE DIAGONAL1 TENSOR
//    size_t nbytes_diagonal1 = sizeof(float) * EMBEDDING_SIZE * 50000;
//    const int64_t diagonal1_dims[] = {50000,EMBEDDING_SIZE};
//    int diag1Size = sizeof(diagonal1_dims)/sizeof(*diagonal1_dims);
//    TF_Tensor* diagonal1_tensor = TF_NewTensor(TF_FLOAT, diagonal1_dims, diag1Size, tensors.diagonals_grads, nbytes_diagonal1, &NoOpDeallocator, 0);
//    //std::cout << "Output info: " << TF_Dim(diagonal1_tensor, 0)  << "  -  " <<  TF_NumDims(diagonal1_tensor) << "  -  " << TF_TensorElementCount(diagonal1_tensor) << "  -  " << TF_TensorType(diagonal1_tensor) <<"\n";
//    
//    
//    vector<TF_Tensor*> final_gradients;
//    final_gradients.push_back(targets_tensor);
//    final_gradients.push_back(targets_gradients_tensor);
//    final_gradients.push_back(contexts_tensor);
//    final_gradients.push_back(contexts_gradients_tensor);
//    final_gradients.push_back(diagonal0_tensor);
//    final_gradients.push_back(diagonal1_tensor);
//    
//    return final_gradients;
//}
//
//
//Mini_Batch_old Model::read_MiniBatch(message_ptr message){
//    int offset = 0;
//    Mini_Batch_old ubatch = Serialization::read_front<Mini_Batch_old>(message.get(), offset);
//    ubatch.target = (int*) malloc(sizeof(int)*mini_batch_size);
//    ubatch.label = (int*) malloc(sizeof(int)*mini_batch_size*TRAIN_EXAMPLES);
//    ubatch.context = (int*) malloc(sizeof(int)*mini_batch_size*TRAIN_EXAMPLES);
//    
//    offset += sizeof(Mini_Batch_old);
//    memcpy(&ubatch.target[0], message->buffer+ offset, sizeof(int)*mini_batch_size);
//    offset += sizeof(int) * mini_batch_size;
//    memcpy(&ubatch.label[0], message->buffer+ offset, sizeof(int)*mini_batch_size*TRAIN_EXAMPLES);
//    offset += sizeof(int) * mini_batch_size * TRAIN_EXAMPLES;
//    memcpy(&ubatch.context[0], message->buffer+ offset, sizeof(int)*mini_batch_size*TRAIN_EXAMPLES);
//    
//    return move(ubatch);
//}
//
//pair<map<int,vector<float>>,map<int,vector<float>>> Model::sumGradientsWindow(pair<map<int,vector<float>>,map<int,vector<float>>> gradients, pair<map<int,vector<float>>,map<int,vector<float>>> gradientsWindow){
//    //gradient values
//    /*
//    int* target_vals = static_cast<int*>(TF_TensorData(gradients[1]));
//    float* target_grad_vals = static_cast<float*>(TF_TensorData(gradients[2]));
//    int* context_vals = static_cast<int*>(TF_TensorData(gradients[3]));
//    float* context_grad_vals = static_cast<float*>(TF_TensorData(gradients[4]));
//    */
//    
//    
//    map<int,vector<float>> targets = gradients.first;
//    map<int,vector<float>> contexts = gradients.second;
//
//    //sum target gradients
//    for (auto const& [key, val] : targets){
//        int target = key;
//        //check if target is already mapped
//        if(gradientsWindow.first.count(target) == 0){ //gradient not in map
//            gradientsWindow.first[target] = val;
//        }else{  //gradient in the map
//            transform(gradientsWindow.first[target].begin(), gradientsWindow.first[target].end(), val.begin(), gradientsWindow.first[target].begin(), std::plus<float>()); //sum gradients
//        }
//    }
//    
//    //sum context gradients
//    for (auto const& [key, val] : contexts){
//        int context = key;
//        //check if target is already mapped
//        if(gradientsWindow.second.count(context) == 0){ //gradient not in map
//            gradientsWindow.second[context] = val;
//        }else{  //gradient in the map
//            transform(gradientsWindow.second[context].begin(), gradientsWindow.second[context].end(), val.begin(), gradientsWindow.second[context].begin(), std::plus<float>()); //sum gradients
//        }
//    }
//    
//    return gradientsWindow;
//}
//
//
//pair<map<int,vector<float>>,map<int,vector<float>>> Model::sumGradients(vector<TF_Tensor*> gradients){
//    //gradient values
//    int* target_vals = static_cast<int*>(TF_TensorData(gradients[2]));
//    float* target_grad_vals = static_cast<float*>(TF_TensorData(gradients[3]));
//    int* context_vals = static_cast<int*>(TF_TensorData(gradients[4]));
//    float* context_grad_vals = static_cast<float*>(TF_TensorData(gradients[5]));
//    
//    map<int,vector<float>> targets;
//    map<int,vector<float>> contexts;
//    
//    //sum target gradients
//    for(int i=0; i < mini_batch_size; i++){
//        int target = target_vals[i];
//        //check if target is already mapped
//        if(targets.count(target) == 0){ //gradient not in map
//            targets[target] = vector<float>(target_grad_vals + (i*EMBEDDING_SIZE), target_grad_vals + ((i+1) * EMBEDDING_SIZE)); //add gradient in map
//        }else{  //gradient in the map
//            vector<float> gradient(target_grad_vals + (i*EMBEDDING_SIZE), target_grad_vals + ((i+1) * EMBEDDING_SIZE));
//            transform(targets[target].begin(), targets[target].end(), gradient.begin(), targets[target].begin(), std::plus<float>()); //sum gradients
//        }
//    }
//    
//    //sum context gradients
//    for(int i=0; i < mini_batch_size*(TRAIN_EXAMPLES); i++){
//        int context = context_vals[i];
//        //check if context is already mapped
//        if(contexts.count(context) == 0){ //gradient not in map
//            contexts[context] = vector<float>(context_grad_vals + (i*EMBEDDING_SIZE), context_grad_vals + ((i+1) * EMBEDDING_SIZE));
//        }else{  //gradient in the map
//            vector<float> gradient(context_grad_vals + (i*EMBEDDING_SIZE), context_grad_vals + ((i+1) * EMBEDDING_SIZE));
//            transform(contexts[context].begin(), contexts[context].end(), gradient.begin(), contexts[context].begin(), std::plus<float>()); //sum gradients
//        }
//    }
//    
//    return make_pair(targets, contexts);
//}
//
//message_ptr Model::constructMessage_Gradients(vector<TF_Tensor*> out_tensors){
//    //create message
//    int n_gradients = (1 + TRAIN_EXAMPLES) * mini_batch_size;
//    message_ptr message = createMessage((n_gradients * sizeof(int)) + (n_gradients * sizeof(float) * EMBEDDING_SIZE) + (sizeof(float)*50000*EMBEDDING_SIZE * 2));
//    
//    //fill message
//    int* targets = static_cast<int*>(TF_TensorData(out_tensors[2]));
//    float* targets_gradients = static_cast<float*>(TF_TensorData(out_tensors[3]));
//    int* contexts = static_cast<int*>(TF_TensorData(out_tensors[4]));
//    float* context_gradients = static_cast<float*>(TF_TensorData(out_tensors[5]));
//    float* diagonals0 = static_cast<float*>(TF_TensorData(out_tensors[6]));
//    float* diagonals1 = static_cast<float*>(TF_TensorData(out_tensors[7]));
//    
//    
//    Serialization::dynamic_event_wrap<int>(targets[0], message.get(), sizeof(int)*mini_batch_size); //wrap target indices
//    Serialization::dynamic_event_wrap<float>(targets_gradients[0], message.get(), sizeof(float)*mini_batch_size*EMBEDDING_SIZE);
//    Serialization::dynamic_event_wrap<int>(contexts[0], message.get(), sizeof(int)*mini_batch_size*TRAIN_EXAMPLES);
//    Serialization::dynamic_event_wrap<float>(context_gradients[0], message.get(), sizeof(float)*mini_batch_size*TRAIN_EXAMPLES*EMBEDDING_SIZE);
//    Serialization::dynamic_event_wrap<float>(diagonals0[0], message.get(), sizeof(float)*50000*EMBEDDING_SIZE);
//    Serialization::dynamic_event_wrap<float>(diagonals1[0], message.get(), sizeof(float)*50000*EMBEDDING_SIZE);
//    
//    float loss_value = static_cast<float*>(TF_TensorData(out_tensors[0]))[0];
//    float acc_value = static_cast<float*>(TF_TensorData(out_tensors[1]))[0];
//    
//    // add loss, acc, model_version, baseline, and #targets
//    Serialization::wrap<float>(loss_value, message.get());
//    Serialization::wrap<float>(acc_value, message.get());
//    Serialization::wrap<unsigned long int>(this->models_version[this->rank], message.get());
//    Serialization::wrap<char>(getBaseline(), message.get());
//    Serialization::wrap<int>(mini_batch_size, message.get());
//    
//    return message;
//}
//
//vector<output_data> Model::constructMessage_Window(pair<map<int,vector<float>>,map<int,vector<float>>> gradients, float* mean_loss, float* mean_acc){
//    map<int,vector<float>> targets = gradients.first;
//    map<int,vector<float>> contexts = gradients.second;
//    
//        
//    //create message
//    int n_gradients = targets.size() + contexts.size(); //size of targets + contexts
//    message_ptr message = createMessage((n_gradients * sizeof(int)) + (n_gradients * sizeof(float) * EMBEDDING_SIZE));
//    
//    //fill message
//    for (auto const& [key, val] : targets){
//        Serialization::wrap<int>(key, message.get()); //wrap target indices
//    }
//    for (auto const& [key, val] : targets){
//        Serialization::dynamic_event_wrap<float>(val[0], message.get(), sizeof(float)*EMBEDDING_SIZE); //wrap target indices
//    }
//    for (auto const& [key, val] : contexts){
//        Serialization::wrap<int>(key, message.get()); //wrap target indices
//    }
//    for (auto const& [key, val] : contexts){
//        Serialization::dynamic_event_wrap<float>(val[0], message.get(), sizeof(float)*EMBEDDING_SIZE); //wrap target indices
//    }
//    
//    
//    float loss_value = *mean_loss;
//    float acc_value = *mean_acc;
//    
//     // add loss, model_version, baseline, and #targets
//     Serialization::wrap<float>(loss_value, message.get());
//     Serialization::wrap<float>(acc_value, message.get());
//     Serialization::wrap<unsigned long int>(this->models_version[this->rank], message.get());
//     Serialization::wrap<char>(getBaseline(), message.get());
//     Serialization::wrap<int>(targets.size(), message.get());
//    
//    
//    // add message to output
//    destination dest = vector<int>({target_all_ranks});
//    vector<output_data> res;
//    res.push_back(make_pair(move(message), dest));
//    
//    return res;
//}
//
//
//
//message_ptr Model::constructMessage_SumGradients(vector<TF_Tensor*> out_tensors){
//    pair<map<int,vector<float>>,map<int,vector<float>>> gradients = sumGradients(out_tensors);
//    map<int,vector<float>> targets = gradients.first;
//    map<int,vector<float>> contexts = gradients.second;
//    
//        
//    //create message
//    int n_gradients = targets.size() + contexts.size(); //size of targets + contexts
//    message_ptr message = createMessage((n_gradients * sizeof(int)) + (n_gradients * sizeof(float) * EMBEDDING_SIZE));
//    
//    //fill message
//    for (auto const& [key, val] : targets){
//        Serialization::wrap<int>(key, message.get()); //wrap target indices
//    }
//    for (auto const& [key, val] : targets){
//        Serialization::dynamic_event_wrap<float>(val[0], message.get(), sizeof(float)*EMBEDDING_SIZE); //wrap target indices
//    }
//    for (auto const& [key, val] : contexts){
//        Serialization::wrap<int>(key, message.get()); //wrap target indices
//    }
//    for (auto const& [key, val] : contexts){
//        Serialization::dynamic_event_wrap<float>(val[0], message.get(), sizeof(float)*EMBEDDING_SIZE); //wrap target indices
//    }
//    
//    
//    float loss_value = static_cast<float*>(TF_TensorData(out_tensors[0]))[0];
//    float acc_value = static_cast<float*>(TF_TensorData(out_tensors[1]))[0];
//    
//     // add loss, acc, model_version, baseline, and #targets
//     Serialization::wrap<float>(loss_value, message.get());
//     Serialization::wrap<float>(acc_value, message.get());
//     Serialization::wrap<unsigned long int>(this->models_version[this->rank], message.get());
//     Serialization::wrap<char>(getBaseline(), message.get());
//     Serialization::wrap<int>(targets.size(), message.get());
//    
//    return message;
//}
//
//pair<map<int,vector<float>>,map<int,vector<float>>> Model::processGradientWindow(message_ptr message, pair<map<int,vector<float>>,map<int,vector<float>>> gradients, float* mean_loss, float* mean_acc){
//    Mini_Batch_old mbatch = read_MiniBatch(move(message));
//    vector<TF_Tensor*> out_tensors = gradient_calc(mbatch);
//    gradients = sumGradientsWindow(move(sumGradients(out_tensors)), gradients);
//    
//    *mean_loss += static_cast<float*>(TF_TensorData(out_tensors[0]))[0] / batch_window_size;
//    *mean_acc += static_cast<float*>(TF_TensorData(out_tensors[1]))[0] / batch_window_size;
//    
//    for(int i = 0; i < out_tensors.size(); i++){
//        TF_DeleteTensor(out_tensors[i]);
//    }
//    
//    free(mbatch.target);
//    free(mbatch.label);
//    free(mbatch.context);
//    
//    
//    return gradients;
//}
//
//
//vector<output_data> Model::processGradientCalc(message_ptr message){
//    
//    Mini_Batch_old mbatch = read_MiniBatch(move(message));
//
//    vector<TF_Tensor*> out_tensors = gradient_calc(mbatch);
//    //sleep(8);
//    
//    // DO NOT SUM GRADIENTS
//    message_ptr message_out = constructMessage_Gradients(out_tensors);
//    
//    //SUM GRADIENTS
//    //message_ptr message_out = constructMessage_SumGradients(out_tensors);
//    
//    
//    //free tensors
//    for(int i = 0; i < out_tensors.size(); i++){
//        TF_DeleteTensor(out_tensors[i]);
//    }
//    
//    // add message to output
//    destination dest = vector<int>({target_all_ranks});
//    vector<output_data> res;
//    res.push_back(make_pair(move(message_out), dest));
//    
//    
//    free(mbatch.target);
//    free(mbatch.label);
//    free(mbatch.context);
//    
//    return res;
//}
//
//void Model::processGradientApplication(message_ptr message){
//    float loss = readLoss(message);
//    float acc = readAcc(message);
//    avg_loss += loss;
//    avg_acc += acc;
//    
//    //read Message
//    int nGradients = (message->size - this->getHeaderSize() - (sizeof(float)*EMBEDDING_SIZE*50000*2)) / (sizeof(int) + (sizeof(float)*EMBEDDING_SIZE));
//    
//    Tensors_str tensors;
//    tensors.nTargets = number_of_targets(message);
//    tensors.nContexts = nGradients - tensors.nTargets;
//    
//    
//    int offset = 0;
//    tensors.targets = (int*) (message->buffer + offset);
//    offset += sizeof(int) * tensors.nTargets ;
//    tensors.target_grads = (float*) (message->buffer + offset);
//    offset += sizeof(float) * tensors.nTargets  * EMBEDDING_SIZE;
//    tensors.contexts = (int*) (message->buffer + offset);
//    offset += sizeof(int) * tensors.nContexts;
//    tensors.context_grads = (float*) (message->buffer + offset);
//    offset += sizeof(float) * tensors.nContexts * EMBEDDING_SIZE;
//    tensors.diagonals = (float*) (message->buffer + offset);
//    offset += sizeof(float) * EMBEDDING_SIZE * 50000;
//    tensors.diagonals_grads = (float*) (message->buffer + offset);
//    
//    vector<TF_Tensor*> grads = recreateTensors(tensors, loss, acc);
//    
//    apply_gradient(grads);
//    
//    unsigned long int old_model_version = readModelVersion(message);
//    unsigned long int new_model_version = this->models_version[this->rank] + batch_window_size;
//    this->models_version[this->rank] = new_model_version;
//        
//    if(rank == 0)
//        progress_bar(epoch, new_model_version);
//    
//    if (new_model_version % (DS_SIZE/mini_batch_size) == 0 && rank == 0){
//        std::cout << std::endl;
//        time_t end;
//        time(&end);
//        cout << "Time end epoch "  << epoch << ": " << fixed
//                 << end;
//            cout << " sec " << endl;
//        epoch++;
//        avg_loss = avg_loss/(DS_SIZE/mini_batch_size);
//        avg_acc = avg_acc/(DS_SIZE/mini_batch_size);
//        std::cout << "Rank:" << rank << ", Loss: " << avg_loss << ", Acc: " << avg_acc << "     - Spoilness:" << new_model_version - old_model_version   << "     - New model Version:" << new_model_version   << "\n";
//        avg_loss= 0;
//        avg_acc = 0;
//        evaluation(evaluation_batches);
//        
//        if(new_model_version / (DS_SIZE/mini_batch_size) == this->epochs){
//            	MPI_Abort(COMM_WORLD, 1);
//		//std::exit(0);
//        }
//        //save("");
//    }
//
//}
//
//void Model::progress_bar(int epoch, int new_model_version){
//    int dataset_size = (DS_SIZE/mini_batch_size);
//    
//    float new_progress = (float)new_model_version / dataset_size;
//    
//    //update progress every 5%
//    if((int)(new_progress*100)%5 != 0 || new_progress - 0.05 <= progress){
//        return;
//    }
//    progress = new_progress;
//    
//    float percentageCompleted = progress - epoch;
//    int barWidth = 70;
//
//    std::cout << "[";
//    int pos = barWidth * percentageCompleted;
//    for (int i = 0; i < barWidth; ++i) {
//        if (i < pos) std::cout << "=";
//        else if (i == pos) std::cout << ">";
//        else std::cout << " ";
//    }
//    std::cout << "] " << new_model_version << "  -  " << int(percentageCompleted * 100.0) << " %\r";
//    std::cout.flush();
//    
//}
