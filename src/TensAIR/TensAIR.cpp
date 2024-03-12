
#include "TensAIR.hpp"
#include <memory>
#include <cassert>
#include <numeric>
#include <unistd.h>
#include <fstream>
#include <cstdlib>
#include <iostream>

/**
 * Default Constructor
 * 
 * tag is the number of this operator in the AIR dataflow
 * rank is the number of this rank on MPI
 * worldSize is the total number of ranks on MPI
 * windowSize is the maximum message size received by TensAIR (either from the EventGenerator or TensAIR itself) <- max(Mini_Batch, Tensor_Data)
 * broadcast_frequency is the number of gradients that shall be calculated before broadcasting them with the other ranks (UNDER DEVELOPMENT, USE 1)
 * epochs is the total number of epochs before stopping dataflow
 * gpus_per_node is the total number of CPUs that are available per node
 * saved_model_dir is the model.tf directory
 * eval_data_file is a binary file containing data to be used for evaluation of the model (formatted with the Mini_Batch struct format)
 * tags is the tensorflow tag used to run the TensorFlow methods (usually "serve")
 * epoch_size is the number of mini_batches that indicate 1 epoch
 * pre_allocate_tensors indicates if tensors shall be pre_allocated (it is necessary to define a fix mini_batch_size) (1=true, 0=false)
 * mini_batch_size is a fixed value that shall be used during training and inferences
 * comm is the MPI object received when using the Python Interface
 * 
 * IMPORTANT: the constructor calls load_tensors(). This method cannot identify the dimention of tensors that vary based on a variable other than mini_batch size. If this is the case, manually specify the tensor dimention after initializing the TensAIR object (before training/evaluating)
 */
TensAIR::TensAIR(const int tag, const int rank, const int worldSize, int windowSize, int broadcast_frequency, int epochs, int gpus_per_node, const char* saved_model_dir, const char* eval_data_file, const char* tags, int epoch_size, float convergence_factor, int epochs_for_convergence, TensAIR::Drift_Mode drift_detector_mode, std::string print_to_folder, int print_frequency, bool preallocate_tensors, int mini_batch_size, MPI_Comm comm) :
BasicVertex<>(tag, rank, worldSize, windowSize, comm) {
    this->gpus_per_node = gpus_per_node;
    this->epochs = epochs;
    this->broadcast_frequency = broadcast_frequency;
    this->epoch_size = epoch_size;
    this->drift_detector_mode = drift_detector_mode;
    this->print_frequency=print_frequency;
    this->drift_identifier_ranks = {this->worldSize}; 
    this->convergence_factor=convergence_factor;
    this->epochs_for_convergence=epochs_for_convergence;
    this->preallocate_tensors=preallocate_tensors;
    this->mini_batch_size=mini_batch_size;


    dest_broadcast = vector<int>({target_all_ranks});
    dest_broadcast.erase(dest_broadcast.begin() + rank); //remove current rank

    //Open file to print results (one file per rank)
    if (print_to_folder.empty()){
        char* path_value = std::getenv("TENSAIR_PATH");
        print_to_folder =  std::string(path_value) + "/output/";
    }
    this->print_to_file = print_to_folder + to_string(rank) + ".csv";
    file_to_print = std::ofstream(print_to_file);


    //Define which GPU each operator must use
    if (gpus_per_node > 0){
        setenv( "CUDA_VISIBLE_DEVICES", to_string(rank%gpus_per_node).c_str(), 1 );
    }else{
        setenv( "CUDA_VISIBLE_DEVICES", "-1", 1 ); //No Gpu
    }
    
    //load TF model
    this->graph = TF_NewGraph();
    this->status = TF_NewStatus();
    this->SessionOpts = TF_NewSessionOptions();
    this->RunOpts = NULL;
    int ntags = 1;
    this->session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, graph, NULL, status);
    

    //check if model was loaded succesfully
    if(TF_GetCode(status) == TF_OK){
        fprintf(stdout, "TF_LoadSessionFromSavedModel OK\n");
    }
    else{
        throw TF_Message(status);
    }
    //load tensors characteristics (names, dimentions, types, etc)
    load_tensors(saved_model_dir, tags);
    

    //pre-allocate input tensors that depend on the mini_batch_size
    if (this->preallocate_tensors){
        if (this->mini_batch_size > 0){
            pre_allocate_tensors(mini_batch_size);
        }else{
            this->preallocate_tensors = false;
            cout << "In order to pre-allocate tensors, it is necessary to define the mini_batch_size. \n Pre-allocation disabled!" <<endl;
        }
    }

    //pre-allocate input tensors that do not depend on the mini_batch_size
    pre_allocate_base_tensors();
    
    //init models_version
    this->models_version = (unsigned long int*) malloc(sizeof(unsigned long int) * worldSize);
    for(int i = 0; i < worldSize; i++){
        this->models_version[i] = (unsigned long int)(0);
    }
    
    //init metrics
    for(int i = 0; i < num_output_metrics; i++){
        metrics_epoch_values.push_back(0);
        metrics_epochs_count.push_back(0);
    }
    
    //try to load evaluation batches
    if (eval_data_file[0] != '\0'){
        try{
            this->evaluation_batches = loadEvaluationBatches(eval_data_file);
            cout << "Evaluation batches loaded successfully!\n";
        }catch(...){
            cout << "\nERROR creating evaluation batches!\n";
        }
    }

}


TensAIR::~TensAIR() {
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    if(print_to_file.empty() == false){
        file_to_print.close();
    }

}

/*
* Static method that fills input and output tensor details 
* 
* parses saved_model_cli string output and store parsed result in the pointers received as input.
* Note: all outputs have the same name, which the method returns as a string and must be set (on the respective class variable) by the method that receives it.
*/
string TensAIR::process_tensor_cli(string cli_output, vector<char*> &tensors, vector<vector<int64_t>> &tensor_dim, vector<TF_DataType> &tensor_dt){
    string nextTensor;

    //check if extracting input or output tensors
    if (cli_output.find("inputs[") !=  string::npos){
        nextTensor = "inputs[";
    }else{
        nextTensor = "outputs[";
    }
    
    int position = 0;
    int start;
    int end;
    int end2;
    int count = 0;
    string term;
    
    string output = "";
    
    while(cli_output.find(nextTensor, position) !=  string::npos){
        //find dtype
        start = (int)cli_output.find("dtype: ", position)+7;
        end = (int)cli_output.find("\n",start);
        term = cli_output.substr(start, end-start);
        TF_DataType dt;
        if(term == "DT_INT32"){
            dt = TF_INT32;
        }else if(term == "DT_FLOAT"){
            dt = TF_FLOAT;
        }else if(term == "DT_INT64"){
            dt = TF_INT64;
        }else if(term == "DT_INT16"){
            dt = TF_INT16;
        }else if(term == "DT_INT8"){
            dt = TF_INT8;
        }else{
            cout << "\n\nERROR - Variable type not supported: " << term << endl;
        }
        tensor_dt.push_back(dt);
        position = end;
        
        //find shape
        start = (int)cli_output.find("shape: (", position)+8;
        end = (int)min(cli_output.find(",",start), cli_output.find(")",start));
        int end_dim = (int)cli_output.find("name: ", position);
        vector<int64_t> dim;
        while(end < end_dim && end != string::npos){
            term = cli_output.substr(start, end-start);
            term = term.length() == 0 ? "1" : term; // solution to empty dimention (1 single number)
            dim.push_back((int64_t)stoi(term));
            start = end + 1; //jump space
            end = (int)min(cli_output.find(",",start), cli_output.find(")",start));
        }
        position = start;
        tensor_dim.push_back(dim);
        
        //find name
        start = (int)cli_output.find("name: ", position)+6;
        end = (int)cli_output.find(":",start);
        term = cli_output.substr(start, end-start);
        if(nextTensor == "inputs["){
            char* inp = (char*)malloc(sizeof(char)*(term.length()+1));
            strcpy(inp, term.c_str());
            tensors.push_back(inp);
        }else if(count == 0){
            output = term;
            count++;
        }
        position = end;
    }
    
    return output;
}

/*
* Method that loads metadata information from input and output tensor
* 
* tensors trackes:
*   - gradient_calc
*   - apply_grad
*   - evaluate
*   - save
*   - prediction
* Note: If the tensors dimentions depend on external variables (other than mini_batch_size), they cannot be directly identified using saved_model_cli. Then, it is necessary to override this and define them manually
*/
void TensAIR::load_tensors(const char *saved_model_dir, const char* tags){
    string command;
    string output;

    //find saved_model_cli location in the system
    
    command = "which saved_model_cli";
    output = exec_shell_command(command.c_str(), 0); //execute command on shell
    output.pop_back(); //remove new line character
    string cli = output;    
    //cli = ".../saved_model_cli";
    cli = SAVED_MODEL_CLI_PATH;

     
    string input;
    string tens_out;
    int startOutput;
    
    //apply_grad
    command = cli + " show --dir " + string(saved_model_dir) + " --tag_set " + string(tags) + " --signature_def apply_gradient";
    output = exec_shell_command(command.c_str(), 0);
    startOutput = (int)output.find("the following output(s):");
    input = output.substr(0, startOutput);
    process_tensor_cli(input, apply_gradient_input, apply_gradient_input_dims, apply_gradient_input_dt);
    output = output.substr(startOutput, output.length()-startOutput);
    vector<char*> apply_gradient_outputs;
    tens_out = process_tensor_cli(output, apply_gradient_outputs, apply_gradient_output_dims, apply_gradient_output_dt);
    apply_gradient_output = (char*)malloc(sizeof(char)*(tens_out.length()+1));
    strcpy(apply_gradient_output, tens_out.c_str());
    
    //evaluate
    command = cli + " show --dir " + string(saved_model_dir) + " --tag_set " + string(tags) + " --signature_def evaluate";
    output = exec_shell_command(command.c_str(), 0);
    startOutput = (int)output.find("the following output(s):");
    input = output.substr(0, startOutput);
    process_tensor_cli(input, evaluate_input, evaluate_input_dims, evaluate_input_dt);
    output = output.substr(startOutput, output.length()-startOutput);
    vector<char*> evaluate_outputs;
    tens_out = process_tensor_cli(output, evaluate_outputs, evaluate_output_dims, evaluate_output_dt);
    evaluate_output = (char*)malloc(sizeof(char)*(tens_out.length()+1));
    strcpy(evaluate_output, tens_out.c_str());
    
    
    //save
    command = cli + " show --dir " + string(saved_model_dir) + " --tag_set " + string(tags) + " --signature_def save";
    output = exec_shell_command(command.c_str(), 0);
    startOutput = (int)output.find("the following output(s):");
    input = output.substr(0, startOutput);
    process_tensor_cli(input, save_input, save_input_dims, save_input_dt);
    output = output.substr(startOutput, output.length()-startOutput);
    vector<char*> save_outputs;
    tens_out = process_tensor_cli(output, save_outputs, save_output_dims, save_output_dt);
    save_output = (char*)malloc(sizeof(char)*(tens_out.length()+1));
    strcpy(save_output, tens_out.c_str());
        
    //prediction
    command = cli + " show --dir " + string(saved_model_dir) + " --tag_set " + string(tags) + " --signature_def prediction";
    output = exec_shell_command(command.c_str(), 0);
    startOutput = (int)output.find("the following output(s):");
    input = output.substr(0, startOutput);
    process_tensor_cli(input, predict_input, predict_input_dims, predict_input_dt);
    output = output.substr(startOutput, output.length()-startOutput);
    vector<char*> predict_outputs;
    tens_out = process_tensor_cli(output, predict_outputs, predict_output_dims, predict_output_dt);
    predict_output = (char*)malloc(sizeof(char)*(tens_out.length()+1));
    strcpy(predict_output, tens_out.c_str());
    
    //clear_delta
    command = cli + " show --dir " + string(saved_model_dir) + " --tag_set " + string(tags) + " --signature_def clear_delta";
    output = exec_shell_command(command.c_str(), 0);
    startOutput = (int)output.find("the following output(s):");
    input = output.substr(0, startOutput);
    process_tensor_cli(input, clear_delta_input, clear_delta_input_dims, clear_delta_input_dt);
    output = output.substr(startOutput, output.length()-startOutput);
    vector<char*> clear_delta_outputs;
    tens_out = process_tensor_cli(output, clear_delta_outputs, clear_delta_output_dims, clear_delta_output_dt);
    clear_delta_output = (char*)malloc(sizeof(char)*(tens_out.length()+1));
    strcpy(clear_delta_output, tens_out.c_str());
    
    //train_step
    command = cli + " show --dir " + string(saved_model_dir) + " --tag_set " + string(tags) + " --signature_def train_step";
    output = exec_shell_command(command.c_str(), 0);
    startOutput = (int)output.find("the following output(s):");
    input = output.substr(0, startOutput);
    process_tensor_cli(input, train_step_input, train_step_input_dims, train_step_input_dt);
    output = output.substr(startOutput, output.length()-startOutput);
    vector<char*> train_step_outputs;
    tens_out = process_tensor_cli(output, train_step_outputs, train_step_output_dims, train_step_output_dt);
    train_step_output = (char*)malloc(sizeof(char)*(tens_out.length()+1));
    strcpy(train_step_output, tens_out.c_str());
    
    //retrieve_delta
    command = cli + " show --dir " + string(saved_model_dir) + " --tag_set " + string(tags) + " --signature_def retrieve_delta";
    output = exec_shell_command(command.c_str(), 0);
    startOutput = (int)output.find("the following output(s):");
    input = output.substr(0, startOutput);
    process_tensor_cli(input, retrieve_delta_input, retrieve_delta_input_dims, retrieve_delta_input_dt);
    output = output.substr(startOutput, output.length()-startOutput);
    vector<char*> retrieve_delta_outputs;
    tens_out = process_tensor_cli(output, retrieve_delta_outputs, retrieve_delta_output_dims, retrieve_delta_output_dt);
    retrieve_delta_output = (char*)malloc(sizeof(char)*(tens_out.length()+1));
    strcpy(retrieve_delta_output, tens_out.c_str());
    
    cout << "Tensors loaded Successfully!" << endl;
    
    return;
}


void TensAIR::pre_allocate_base_tensors(){
    ///////apply gradient
    pair<vector<TF_Output>, vector<TF_Tensor*>> result = allocateTensor(mini_batch_size, apply_gradient_input, apply_gradient_input_dims, apply_gradient_input_dt);
    app_inp_op = result.first;
    app_inp_tensors = result.second;
    n_app_inp = (int)apply_gradient_input_dims.size();
    //parse to pointers instead of vectors
    inp_app_operators = (TF_Output*) malloc(sizeof(TF_Output)*n_app_inp);
    inp_app_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*n_app_inp);
    //copy pointers (not necessary to copy the data)
    for (int i = 0; i < n_app_inp; i ++){
        inp_app_operators[i] = app_inp_op[i];
        inp_app_values[i] = app_inp_tensors[i];
    }

    ///////clear delta
    result = allocateTensor(1, clear_delta_input, clear_delta_input_dims, clear_delta_input_dt);
    clear_delta_inp_op = result.first;
    clear_delta_inp_tensors = result.second;
    n_clear_delta_inp = (int)clear_delta_input_dims.size();
    //parse to pointers instead of vecotrs
    clear_delta_operators = (TF_Output*) malloc(sizeof(TF_Output)*n_clear_delta_inp);
    clear_delta_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*n_clear_delta_inp);
    //copy pointers (not necessary to copy the data)
    for (int i = 0; i < n_clear_delta_inp; i ++){
        clear_delta_operators[i] = clear_delta_inp_op[i];
        clear_delta_values[i] = clear_delta_inp_tensors[i];
    }

     ///////retrieve_delta
    result = allocateTensor(1, retrieve_delta_input, retrieve_delta_input_dims, retrieve_delta_input_dt);
    retrieve_delta_inp_op = result.first;
    retrieve_delta_inp_tensors = result.second;
    n_retrieve_delta_inp = (int)retrieve_delta_input_dims.size();
    //parse to pointers instead of vecotrs
    retrieve_delta_operators = (TF_Output*) malloc(sizeof(TF_Output)*n_retrieve_delta_inp);
    retrieve_delta_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*n_retrieve_delta_inp);
    //copy pointers (not necessary to copy the data)
    for (int i = 0; i < n_retrieve_delta_inp; i ++){
        retrieve_delta_operators[i] = retrieve_delta_inp_op[i];
        retrieve_delta_values[i] = retrieve_delta_inp_tensors[i];
    }
}

// preallocate input tensors
void TensAIR::pre_allocate_tensors(int mini_batch_size){

    ///////train_step
    pair<vector<TF_Output>, vector<TF_Tensor*>> result = allocateTensor(mini_batch_size, train_step_input, train_step_input_dims, train_step_input_dt);
    train_step_inp_op = result.first;
    train_step_inp_tensors = result.second;
    n_train_step_inp = (int)train_step_input_dims.size();
    //parse to pointers instead of vecotrs
    train_step_operators = (TF_Output*) malloc(sizeof(TF_Output)*n_train_step_inp);
    train_step_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*n_train_step_inp);
    //copy pointers (not necessary to copy the data)
    for (int i = 0; i < n_train_step_inp; i ++){
        train_step_operators[i] = train_step_inp_op[i];
        train_step_values[i] = train_step_inp_tensors[i];
    }

    ///////predict
    result = allocateTensor(1, predict_input, predict_input_dims, predict_input_dt);
    predict_inp_op = result.first;
    predict_inp_tensors = result.second;
    n_predict_inp = (int)predict_input_dims.size();
    //parse to pointers instead of vecotrs
    predict_operators = (TF_Output*) malloc(sizeof(TF_Output)*n_predict_inp);
    predict_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*n_predict_inp);
    //copy pointers (not necessary to copy the data)
    for (int i = 0; i < n_predict_inp; i ++){
        predict_operators[i] = predict_inp_op[i];
        predict_values[i] = predict_inp_tensors[i];
    }
}

// execute shell command from input and returns the shell output as string
std::string TensAIR::exec_shell_command(const char* cmd, int try_times, int max_try_times) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        if (try_times < max_try_times){
            cout << "Problem running: " << cmd << "\n Trying again... (" << try_times << "/"<< max_try_times << ")\n";
            return exec_shell_command(cmd, try_times++);
        }else{
            throw;
        }
        
    }
    pclose(pipe);
    return result;
}

pair<vector<TF_Output>, vector<TF_Tensor*>> TensAIR::allocateTensor(int mini_batch_size, vector<char*> signature, vector<vector<int64_t>> dims, vector<TF_DataType> dt){
    bool output = false;
    if(signature.size() < dims.size()){
        output = true;
        for(int i = (int)signature.size(); i < (int)dims.size(); i++){
            signature.push_back(signature[0]);
        }
    }
   
    vector<TF_Output> opouts;
    vector<TF_Tensor*> tensors;
    
    //malloc dimentions
    int64_t** dim = (int64_t**)malloc(sizeof(int64_t*)*dims.size());
    for(int i = 0; i < dims.size(); i++){
        dim[i] = (int64_t*)malloc(sizeof(int64_t)*dims[i].size());
    }
    
    //create tensors
    for(int i = 0; i < signature.size(); i++){
        TF_Operation* op = TF_GraphOperationByName(graph, signature[i]);
        TF_Output opout = {op};
        if(output){
            opout.index=i;
        }
        
        //calculate tensor size and dimenstions
        size_t nbytes = 1;
        for(int j = 0; j < dims[i].size(); j++){
            if(dims[i][j] < 0){ //check if dimention is based on mini_batch_size
                //negative dimentions are based on the mini batch and shall be multiplied by the minibatch size
                nbytes *= mini_batch_size * dims[i][j] * -1;
                dim[i][j] = (int64_t)(mini_batch_size * dims[i][j] * -1);
            }else{
                nbytes *= dims[i][j];
                dim[i][j] = (int64_t)dims[i][j];
            }
        }
        
        nbytes *= TF_DataTypeSize(dt[i]);
        int num_dimentions = (int)dims[i].size();
        
        TF_Tensor* tensor = TF_AllocateTensor(dt[i], dim[i], num_dimentions, nbytes);
        //// As with inputs, check the values for the output operation and output tensor
        //std::cout << "Input: " << TF_OperationNumOutputs(op) << "\n";
        //std::cout << "Input info: " << TF_Dim(tensor, 0) << "\n";
        
        opouts.push_back(opout);
        tensors.push_back(tensor);
    }
    
    for(int i =0; i < dims.size(); i++){
        free(dim[i]);
    }
    free(dim);
    
    return make_pair(opouts, tensors);
}


//Main method that manages TensAIR training
void TensAIR::streamProcess(int channel) {
        //////// There are worldSize * 3 channels.
        //      channel < worldSize -> messages received from EventGenerator
        //      worldSize < channel < 2*worldSize -> messages received from other Model
        //      2*worldSize >= channel -> messages received from other Drift Detector

        time(&start);

        if(channel < this->worldSize){ //messages receved from Event Generator. (calculate gradient or prediction)
            list<message_ptr> pthread_waiting_lists;
            while(this->ALIVE){
                //wait until update list is empty and model is up-to-date
                //or
                //model evaluation is not occurring
                while (!model_update_list.empty() || isEvaluating) {
                    pthread_cond_wait(&empty_list_cond, &empty_list_mutex);
                }
                message_ptr message = BasicVertex<>::fetchNextMessage(channel, pthread_waiting_lists);

                bool end = message_from_generator(std::move(message));
                if (end){
                    this->ALIVE=false;
                    break;
                }
                
                
            }

        }else if(channel < this->worldSize * 2){ //messages received from Model (gradient to be applied or a signal of convergence)
            while(this->ALIVE){
                //update model_update_list
                fetchUpdateMessages(channel, model_update_list);
                
                //one thread applies the updates
                if(channel == model_update_rank){
                    message_ptr message = fetchNextMessage(channel, model_update_list);

                    bool end = message_from_model(std::move(message));
                    if (end){
                        this->ALIVE=false;
                        break;
                    }
                    
                }
                
            }
        }else{ //messages received from Concept Drift detector (update drift status)
            while(this->ALIVE){
                //update drift_list
                fetchUpdateMessages(channel, drift_list);
                //one thread applies the updates
                if(channel == model_update_drift_rank){
                    message_ptr message = fetchNextMessage(channel, drift_list);

                    message_from_drift_detector(std::move(message));
                }
            }
        }
}

bool TensAIR::message_from_generator(message_ptr message){
    bool end = false;

    ///////// mini_batch for training or prediction?
    // calculate gradient if model has not converged or passive drift adaptation is enabled
    if ((!has_converged && drift_detector_mode != TensAIR::Drift_Mode::NEVER_TRAIN) || drift_detector_mode == TensAIR::Drift_Mode::ALWAYS_TRAIN){    
        //calculate gradient
        train_step(std::move(message));
        
        if(gradientsCalculated % broadcast_frequency == 0){
            pair<bool,bool> afp = broadcast_gradient();
            bool convergence_reached = afp.first;
            bool end_stream = afp.second;
            
            if (convergence_reached == true){
                message_ptr convergence_message = createMessage(sizeof(int));
                Serialization::wrap<int>(rank, convergence_message.get());
                destination dest = vector<int>({target_all_ranks});
                vector<output_data> res;
                res.push_back(make_pair(std::move(convergence_message), dest));
                this->send_to_specific_operator(std::move(res), 0);
            }
            if (end_stream == true){
                end=true;
            }
            
        }
        
    // calculate prediction if model has converged 
    }else{
        vector<output_data> prediction_loss = predict(std::move(message));
        //if drift detection is enable send loss to drift detector
        if (drift_detector_mode == TensAIR::Drift_Mode::AUTOMATIC){
            this->send_to_specific_operator(std::move(prediction_loss),0); //send loss to concept drift detector
        }
    }
    return end;
}

bool TensAIR::message_from_model(message_ptr message){
    bool end = false;
    //received signal of convergence 
    if(message->size == sizeof(int)){
        if(this->has_converged == false){
            this->has_converged=true;
            this->reset_drtift_detector = true;
            past_metrics.clear();
        }
        return end;
    }
    
    //receive and apply gradient
    pair<bool,bool> afp = apply_gradient(std::move(message));
    
    bool convergence_reached = afp.first;
    bool end_stream = afp.second;


    
    //if convergence is reached signal to other ranks that model converged
    if(convergence_reached){
        message_ptr convergence_message = createMessage(sizeof(int)); 
        Serialization::wrap<int>(rank, convergence_message.get()); 
        destination dest = vector<int>({target_all_ranks});
        vector<output_data> res;
        res.push_back(make_pair(std::move(convergence_message), dest));
        this->send_to_specific_operator(std::move(res), 0);
    }
    
    pthread_cond_signal(&empty_list_cond); //signal to the synchronization among models that the model performed an update
    
    if(end_stream){
        end=true;
    }

    return end;

}

void TensAIR::message_from_drift_detector(message_ptr message){
    int offset = 0;
    this->drift_magnetude = Serialization::read_front<float>(message.get(), offset); //read drift magnetude
    this->has_converged = false;
}

//fetch messages from lis buffer (which arrived from other TensAIR ranks) 
message_ptr TensAIR::fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list){
    pthread_mutex_lock(&update_list_mutex);

    message_ptr message = std::move(pthread_waiting_list.front()); // move
    pthread_waiting_list.pop_front();
    
    pthread_mutex_unlock(&update_list_mutex);

    return message;
}

/*
 * Fetches new messages from the input buffer.
 *     Updates in the model are inserted at the beggining of the list. Then, the train steps are inserted at the end of it.
 */
void TensAIR::fetchUpdateMessages(int channel, list<message_ptr>& pthread_waiting_list){
    pthread_mutex_lock(&this->listenerMutexes[channel]);
    
    //wait until new messages arrive in this or in other channels
    if(channel == model_update_rank){
        while(this->inMessages[channel].empty() && pthread_waiting_list.empty()){
            pthread_cond_signal(&empty_list_cond); //signal that the model has no updates to perform yet.
            pthread_cond_wait(&this->listenerCondVars[channel], &this->listenerMutexes[channel]);
        }
    }else{// wait for messages in this channel
        while(this->inMessages[channel].empty()){
            pthread_cond_signal(&empty_list_cond); //signal that the model has no updates to perform yet.
            pthread_cond_wait(&this->listenerCondVars[channel], &this->listenerMutexes[channel]);
        }
    }
    
    
    // when new messages arrive, add them to the list
    if(!this->inMessages[channel].empty()){
        
        pthread_mutex_lock(&update_list_mutex);
        while(!this->inMessages[channel].empty()){
            message_ptr inMessage(this->inMessages[channel].front());
            pthread_waiting_list.push_back(std::move(inMessage));
            this->inMessages[channel].pop_front();
        }
        pthread_mutex_unlock(&update_list_mutex);
        pthread_cond_signal(&this->listenerCondVars[model_update_rank]); //signal to model_update_rank channel that new messages arrived
    }
    pthread_mutex_unlock(&this->listenerMutexes[channel]);
    return;
}

/*
 * Runs method on TensorFlow C API
 * 
 * input_operator contains name of input tensors
 * input_calues contains Tensors with the input data
 * nInputs is the number of inputs
 * target_operator contains name of output tensors
 * target_value contains name of output tensors
 * nOutputs contains the output Tensors with memory pre-allocated
 */ 
TF_Tensor** TensAIR::runSession(vector<TF_Output> input_operator, vector<TF_Tensor*> input_value, int nInputs, vector<TF_Output> target_operator, vector<TF_Tensor*> target_value, int nOutputs){
    assert(TF_GetCode(status) == TF_OK);
    
    //parse to pointers instead of vecotrs
    TF_Output* inp_operators = (TF_Output*) malloc(sizeof(TF_Output)*nInputs);
    TF_Tensor** inp_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*nInputs);
    //copy pointers (not necessary to copy the data)
    for (int i = 0; i < nInputs; i ++){
        inp_operators[i] = input_operator[i];
        inp_values[i] = input_value[i];
    }
    
    //parse to pointers instead of vecotrs
    TF_Output* tar_operators = (TF_Output*) malloc(sizeof(TF_Output)*nOutputs);
    TF_Tensor** tar_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*nOutputs);
    //copy pointers (not necessary to copy the data)
    for (int i = 0; i < nOutputs; i ++){
        tar_operators[i] = target_operator[i];
        tar_values[i] = target_value[i];
    }
    
    //run C API 
    TF_SessionRun(session, nullptr,
                  inp_operators, inp_values, nInputs,
                  tar_operators, tar_values, nOutputs,
                  nullptr, 0, nullptr, status);
    
    //check for error
    if (TF_GetCode(status) != TF_OK) {
      cout << "ERROR: something wrong with encoding:" << TF_Message(status);
      throw "ERROR: something wrong with encoding: %s", TF_Message(status);
    }
    
    //free memory
    free(inp_operators);
    free(inp_values);
    free(tar_operators);
    
    return tar_values;
    
}

/*
 * construct Mini_Batch struct from message arriving from EventGenerator
 *
 *   MESSAGE FORMAT
 *   mini_batch_size                                                         (int mini_batch_size)
 *   num_inputs                                                              (int num_inputs)
 *   size_input_0, size_inpu_1, ... , size_input_num_inputs                  (int[num_inputs] size_inputs) [note: size in bytes]
 *   input_0, input_1, ... , input_num_inputs                                (char[num_inputs][size_input[num_input]])
 */
Mini_Batch TensAIR::read_MiniBatch(message_ptr message){
    Mini_Batch mbatch;

    int offset = 0;
    mbatch.mini_batch_size = Serialization::read_front<int>(message.get(), offset); //read mini_batch_size
    offset += sizeof(int);
    mbatch.num_inputs = Serialization::read_front<int>(message.get(), offset); //read num_inputs
    offset += sizeof(int);
    mbatch.size_inputs = (size_t*) malloc(sizeof(size_t)*mbatch.num_inputs); //allocate size_inputs based on num_inputs
    memcpy(mbatch.size_inputs, message->buffer+ offset, sizeof(size_t)*mbatch.num_inputs); //read size_inputs
    offset += sizeof(size_t)*mbatch.num_inputs;


    mbatch.inputs = (void**) malloc(sizeof(void*)*mbatch.num_inputs); //allocate inputs list based on num_inputs
    for(int i = 0; i < mbatch.num_inputs; i++){
        mbatch.inputs[i] = (void*)malloc(mbatch.size_inputs[i]); //allocate each input based on size_inputs
        memcpy(mbatch.inputs[i], message->buffer+ offset, mbatch.size_inputs[i]); //read input
        offset += mbatch.size_inputs[i];
    }

    return std::move(mbatch);
}

/*
 * Reads evaluation datatset (divided in batches) from a file
 * 
 * The file should be written on the following format:
 * mini_batch_size num_training_examples num_tensors size_tensor_0 tensor_0 ... size_tensor_n tensor_n
 * num_training_examples, num_tensors, size_tensor_i (in bytes): int
 * tensor_i : size_input_i * bytes
 * IMPORTANT: char** tensors is never deallocated since it will be used for evaluating the model indefinitely!
 */
vector<Mini_Batch> TensAIR::loadEvaluationBatches(const char* file){
    vector<Mini_Batch> evaluation_batches;
    ifstream infile(file, std::ios::binary);

    char buffer[4];
    int num_training_examples;
    int num_tensors;
    int mini_batch_size;
    
    
    //Read File
    infile.read(buffer, 4); //read mini_batch_size
    mini_batch_size = static_cast<int*>(static_cast<void*>(&buffer))[0];
    infile.read(buffer, sizeof(int)); //read num_training_examples
    num_training_examples = static_cast<int*>(static_cast<void*>(&buffer))[0];
    infile.read(buffer, sizeof(int)); //read num_tensors
    num_tensors = static_cast<int*>(static_cast<void*>(&buffer))[0];
    
    char** tensors = (char**)malloc(sizeof(char*)*num_tensors);
    int* size_tensors = (int*)malloc(sizeof(int)*num_tensors);
    
    for(int i = 0; i < num_tensors; i++){
        infile.read(buffer, sizeof(int)); //read tensor_i size
        size_tensors[i] = static_cast<int*>(static_cast<void*>(&buffer))[0];
        tensors[i] = (char*)malloc(size_tensors[i]); //read tensor_i
        infile.read(tensors[i], size_tensors[i]);
    }
    infile.close();
    
    
    // Parse evaluation data to vector of mini_batches
    int number_mini_batches = num_training_examples/mini_batch_size; //define number of mini_batches based on the mini_batch size and number of training examples
    
    for (int i = 0; i < number_mini_batches; i++){
        //create mini_batch
        Mini_Batch evaluation_batch;
        evaluation_batch.mini_batch_size = mini_batch_size;
        evaluation_batch.num_inputs = num_tensors;
        evaluation_batch.size_inputs = (size_t*) malloc(sizeof(size_t)*num_tensors);
        evaluation_batch.inputs = (void**) malloc(sizeof(void*)*num_tensors);
        //fill the tensors of the mini_batch with the data read from the file
        for(int j = 0; j < num_tensors; j++){
            evaluation_batch.size_inputs[j] = (size_t)((size_tensors[j]/num_training_examples)*number_mini_batches); //the tensor size is calculated based on the number of training examples per mini_batch
            evaluation_batch.inputs[j] = static_cast<void*>(&(tensors[j][i * evaluation_batch.size_inputs[j]]));
        }
        evaluation_batches.push_back(evaluation_batch);
    }
    
    free(size_tensors);

    return evaluation_batches;
}

/*
 * serializes Tensors to a message, which willbe send to other TensAIR ranks
 * 
 * MESSAGE1 FORMAT (to send to other models)
 * model_version                                                           (unsigned long int model_version)
 * mini_batch_size                                                         (int mini_batch_size) 
 * num_output_metrics                                                      (int num_output_metrics)
 * num_gradients                                                           (int num_gradients)
 * size_gradient_0, size_gradient_1, ... , size_gradient_num_gradients     (int[num_gradient] size_gradient) [note: size in bytes]
 * gradient_0, gradient_1, ... , gradient_num_gradient                     (char[num_gradients][size_gradient[num_gradient]])
 */
message_ptr TensAIR::construct_Message_Tensors(vector<TF_Tensor*> out_tensors, int num_output_metrics, int mini_batch_size, int n_delta){
    //allocate message gradients
    size_t message_size = sizeof(unsigned long int) + sizeof(int) + sizeof(int) + sizeof(int) + sizeof(int) + (sizeof(size_t)*out_tensors.size());
    for (int i = 0; i < out_tensors.size(); i ++){
        message_size += TF_TensorByteSize(out_tensors[i]); //get size of tensor in bytes
    }
    message_ptr message = createMessage(message_size); //allocate memory for message

    //fill message gradients
    Serialization::wrap<unsigned long int>(this->models_version[this->rank], message.get()); //fill model_version
    Serialization::wrap<int>(mini_batch_size, message.get()); //fill mini_batch_size
    Serialization::wrap<int>(n_delta, message.get()); //fill mini_batch_size
    Serialization::wrap<int>(num_output_metrics, message.get()); //fill num_output_metrics
    Serialization::wrap<int>((int)out_tensors.size()-num_output_metrics, message.get()); //fill num_gradients
    for(int i = 0; i < out_tensors.size(); i++){
        Serialization::wrap<size_t>(TF_TensorByteSize(out_tensors[i]), message.get()); //fill size of gradients
    }
    for(int i = 0; i < out_tensors.size(); i++){
        size_t gradient_size = TF_TensorByteSize(out_tensors[i]);
        char* tensor_data = static_cast<char*>(TF_TensorData(out_tensors[i]));
        Serialization::dynamic_event_wrap<char>(tensor_data[0], message.get(), gradient_size); //fill gradients_data
    }

    return message;
}

/*
 * serializes Tensors to a message, which will be send to drift detector
 * MESSAGE2 FORMAT (to send to drift detector)
 * loss                                                                    (int)
 * reset_detector                                                          (int) 
 */
message_ptr TensAIR::construct_Message_Tensors_Loss(vector<TF_Tensor*> out_tensors, int num_output_metrics){
    //allocate message loss
    size_t message_size = sizeof(float) + sizeof(int);
    message_ptr message = createMessage(message_size); //allocate memory for message

    vector<float> loss(1, 0.0);
    loss[0] += static_cast<float*>(TF_TensorData(out_tensors[0]))[0]; //get loss
    Serialization::wrap<float>(loss[0], message.get()); //fill loss into message
    if (reset_drtift_detector){
        Serialization::wrap<int>(1, message.get()); //set it is necessary to reset_drtift_detector
        reset_drtift_detector = false;
    }else{
        Serialization::wrap<int>(0, message.get());//set it is not necessary to reset_drtift_detector
    }

    return message;
}


//receives message from Event Generator and returns message with Tensors(gradients) (to send to all TensAIR ranks) & loss (to send to drift detector)
void TensAIR::train_step(message_ptr message){
    //deserialize message to Mini_Batch
    Mini_Batch mbatch = read_MiniBatch(std::move(message));
    
    if(preallocate_tensors){
        ///////create output tensors
        vector<char*> train_step_outputs(1,train_step_output);
        pair<vector<TF_Output>, vector<TF_Tensor*>> result = allocateTensor(mini_batch_size, train_step_outputs, train_step_output_dims, train_step_output_dt);
        train_step_out_op = result.first;
        train_step_out_tensors = result.second;
        n_train_step_out = (int)train_step_output_dims.size();
        //parse to pointers instead of vectors
        TF_Output* tar_operators = (TF_Output*) malloc(sizeof(TF_Output)*n_train_step_out);
        TF_Tensor** tar_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*n_train_step_out);
        //copy pointers (not necessary to copy the data)
        for (int i = 0; i < n_train_step_out; i ++){
            tar_operators[i] = train_step_out_op[i];
            tar_values[i] = train_step_out_tensors[i];
        }


        ///////assign minibatch data to input tensors
        copyDataToTensors(train_step_values, mbatch.inputs, mbatch.num_inputs);

        //calculate gradient
        TF_SessionRun(session, nullptr,
                    train_step_operators, train_step_values, n_train_step_inp,
                    tar_operators, tar_values, n_train_step_out,
                    nullptr, 0, nullptr, status);

        //delete output tensors (not values)
        for(int i = 0; i < train_step_out_tensors.size(); i++){
            TF_DeleteTensor(train_step_out_tensors[i]);
        }
        free(tar_operators);
        for(int i = 0; i < n_train_step_out; i++){
            TF_DeleteTensor(tar_values[i]);
        }
        free(tar_values);

    }else{
        ////////////calculate gradients
        vector<TF_Tensor*> out_tensors = run_tf_function(mbatch.inputs, mbatch.mini_batch_size, train_step_input, train_step_output, train_step_input_dims, train_step_output_dims, train_step_input_dt, train_step_output_dt);
        
        for(int i = 0; i < out_tensors.size(); i++){
            TF_DeleteTensor(out_tensors[i]);
        }
    }
    
    //update version
    unsigned long int new_model_version = this->models_version[this->rank] + 1;
    this->models_version[this->rank] = new_model_version;
    gradientsCalculated++;
    local_gradient_applied++;
    
    
    //delete input values
    free(mbatch.size_inputs);
    for(int i = 0; i < mbatch.num_inputs; i++)
        free(mbatch.inputs[i]);
    free(mbatch.inputs);

    return;
}

//receives message from Event Generator and returns message with predictions (to send to next Vertex on dataflow) & loss (to send to drift detector)
vector<output_data> TensAIR::predict(message_ptr message){
    count_predictions++;
    //deserialize message to Mini_Batch
    Mini_Batch mbatch = read_MiniBatch(std::move(message)); 

    //calculate gradient
    vector<TF_Tensor*> out_tensors = run_tf_function(mbatch.inputs, mbatch.mini_batch_size, predict_input, predict_output, predict_input_dims, predict_output_dims, predict_input_dt, predict_output_dt);
    float loss = static_cast<float*>(TF_TensorData(out_tensors[0]))[0];
    float prediction = static_cast<float*>(TF_TensorData(out_tensors[1]))[0];

    //serialize gradients to message
    message_ptr message_out_loss = construct_Message_Tensors_Loss(out_tensors, num_output_metrics);

    // check if results should be printed to file
    if(print_to_file.empty() == false && count_predictions % print_frequency == 0){
        file_to_print << "predicting, " << count_predictions << "," << loss << "," << prediction << std::endl;
    }

    //free data
    for(int i = 0; i < out_tensors.size(); i++){
        TF_DeleteTensor(out_tensors[i]);
    }
    
    

    // add message to output
    destination dest2 = vector<int>({drift_identifier_ranks});
    vector<output_data> res;
    res.push_back(make_pair(std::move(message_out_loss), dest2));

    free(mbatch.size_inputs);
    for(int i = 0; i < mbatch.num_inputs; i++)
        free(mbatch.inputs[i]);
    free(mbatch.inputs);
    return res;
}

/*
 * constructs Tensor_Data struct from message arriving from other TensAIR rank
 *  
 * MESSAGE FORMAT
 * model_version                                                           (unsigned long int model_version)
 * mini_batch_size                                                         (int mini_batch_size)
 * num_output_metrics                                                      (int num_output_metrics)
 * num_tensor                                                              (int num_tensors)
 * size_tensor_0, size_tensor_1, ... , size_tensor_num_tensor              (int[num_tensor] size_tensor) [note: size in bytes]
 * tensor_0, tensor_1, ... , tensor_num_tensor                             (char[num_tensors][size_tensor[num_tensor]])
 */
Tensor_Data TensAIR::read_Tensors(message_ptr message){
    Tensor_Data tens_data;
   
    int offset = 0;
    tens_data.model_version = (int)Serialization::read_front<unsigned long int>(message.get(), offset); //read model_version  
    offset += sizeof(unsigned long int);
    tens_data.mini_batch_size = Serialization::read_front<int>(message.get(), offset); //read mini_batch_size
    offset += sizeof(int);
    tens_data.n_delta = Serialization::read_front<int>(message.get(), offset); //read n_delta
    offset += sizeof(int);
    tens_data.num_output_metrics = Serialization::read_front<int>(message.get(), offset); //read num_output_metrics
    offset += sizeof(int);
    tens_data.num_gradients = Serialization::read_front<int>(message.get(), offset); //read num_gradients
    offset += sizeof(int);

    //read size of metrics (the first tensors are always the metrics)
    tens_data.size_metrics = (size_t*) malloc(sizeof(size_t)*tens_data.num_output_metrics); //allocate size_metrics based on num_output_metrics
    memcpy(tens_data.size_metrics, message->buffer+ offset, sizeof(size_t)*tens_data.num_output_metrics); //read size_metrics
    offset += sizeof(size_t)*tens_data.num_output_metrics;

    //read size of gradients
    tens_data.size_gradients = (size_t*) malloc(sizeof(size_t)*tens_data.num_gradients); //allocate size_gradients based on num_gradients
    memcpy(tens_data.size_gradients, message->buffer+ offset, sizeof(size_t)*tens_data.num_gradients); //read size_gradients
    offset += sizeof(size_t)*tens_data.num_gradients;

    //read metrics (the first tensor are always the metrics)
    tens_data.metrics_data = (float**) malloc(sizeof(float*)*tens_data.num_output_metrics); //allocate metrics list based on num_metrics
    for(int i = 0; i < tens_data.num_output_metrics; i++){
        tens_data.metrics_data[i] = (float*)malloc(tens_data.size_metrics[i]); //allocate each metric based on size_metrics
        memcpy(tens_data.metrics_data[i], message->buffer+ offset, tens_data.size_metrics[i]); //read metric
        offset += tens_data.size_metrics[i];
    }

    //read gradients
    tens_data.gradients_data = (void**) malloc(sizeof(void*)*tens_data.num_gradients); //allocate gradients list based on num_gradients
    for(int i = 0; i < tens_data.num_gradients; i++){
        tens_data.gradients_data[i] = (void*)malloc(tens_data.size_gradients[i]); //allocate each gradinet based on size_gradients
        memcpy(tens_data.gradients_data[i], message->buffer+ offset, tens_data.size_gradients[i]); //read gradient
        offset += tens_data.size_gradients[i];
    }

    return std::move(tens_data);
}


/**
 * Sends multiple messages to one of the next (specific) operators in the dataflow.
 * It is possible to modify which ranks will receive the messages
 * by setting up correctly the target_ranks vector.
 * 
 * It is also possible to define which is the target operator. targetOperator_index = position of the operator in the links defined in the dataflow (in respect to current operator)
 * */
void TensAIR::send_to_specific_operator(vector<output_data> messages, int targetOperator_index){

    for (output_data& data : messages){

        if (data.second.size() == 0){
            throw "[BasicVertex](send) Message has no destination.";
        } else if (data.second.size() > worldSize) {
            throw "[BasicVertex](send) Message destination rank does not exist.";
        } else {
            for(int targetRank : data.second){
                size_t target = targetOperator_index * worldSize + targetRank;

                Message* cpy = Serialization::copy(data.first.get());

                pthread_mutex_lock(&senderMutexes[target]);
                outMessages[target].push_back(cpy);
                pthread_cond_signal(&senderCondVars[target]);
                pthread_mutex_unlock(&senderMutexes[target]);
            }
        }
    }
}

//updates local metrics based on metrics from new gradients (METRICS MUST BE FLOAT)
void TensAIR::update_metrics(int num_output_metrics, float** metrics_data, int n_delta){
    for(int i = 0; i < num_output_metrics; i++){
        metrics_epoch_values[i] += ((float) metrics_data[i][0]);
        metrics_epochs_count[i] += n_delta;
    }
    return;
}

// receives message from other TensAIR rank and apply it locally.
pair<bool,bool> TensAIR::apply_gradient(message_ptr message){
    bool converged = false;
    //deserialize message to Tensor_Data
    Tensor_Data tens_data = read_Tensors(std::move(message));
    
    pair<bool,bool> afp = after_gradient_application(tens_data.metrics_data, num_output_metrics, tens_data.n_delta);

    if(preallocate_tensors){
        ////////apply gradients locally (with the rest of the tensors)
        //create output tensors
        vector<char*> apply_gradient_outputs(1,apply_gradient_output);
        pair<vector<TF_Output>, vector<TF_Tensor*>> result = allocateTensor(mini_batch_size, apply_gradient_outputs, apply_gradient_output_dims, apply_gradient_output_dt);
        app_out_op = result.first;
        app_out_tensors = result.second;
        n_app_out = (int)apply_gradient_output_dims.size();
        //parse to pointers instead of vecotrs
        TF_Output* tar_operators = (TF_Output*) malloc(sizeof(TF_Output)*n_app_out);
        TF_Tensor** tar_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*n_app_out);
        //copy pointers (not necessary to copy the data)
        for (int i = 0; i < n_app_out; i ++){
            tar_operators[i] = app_out_op[i];
            tar_values[i] = app_out_tensors[i];
        }

        //assign minibatch data to input tensor
        copyDataToTensors(inp_app_values, tens_data.gradients_data, n_app_inp);
        
        //apply gradient
        TF_SessionRun(session, nullptr,
                    inp_app_operators, inp_app_values, n_app_inp,
                    tar_operators, tar_values, n_app_out,
                    nullptr, 0, nullptr, status);
        
        
        for(int i = 0; i < app_out_tensors.size(); i++){
            TF_DeleteTensor(app_out_tensors[i]);
            TF_DeleteTensor(tar_values[i]);
        }
        
        free(tar_operators);
    }else{
        //apply gradients locally (with the rest of the tensors)
        vector<TF_Tensor*> output = run_tf_function(tens_data.gradients_data, tens_data.mini_batch_size, apply_gradient_input, apply_gradient_output, apply_gradient_input_dims, apply_gradient_output_dims, apply_gradient_input_dt, apply_gradient_output_dt);

        for(int i = 0; i < output.size(); i++){
            TF_DeleteTensor(output[i]);
        }

    }


    //update version
    unsigned long int old_model_version = tens_data.model_version;
    unsigned long int new_model_version = this->models_version[this->rank] + broadcast_frequency;
    this->models_version[this->rank] = new_model_version;


    for(int i = 0; i < tens_data.num_output_metrics; i++)
        free(tens_data.metrics_data[i]);
    free(tens_data.metrics_data);
    
    for(int i = 0; i < tens_data.num_gradients; i++)
        free(tens_data.gradients_data[i]);
    free(tens_data.gradients_data);
    
    
    
    return afp;
}

// evaluates model using evaluation data received in the constructor
void TensAIR::evaluate(vector<Mini_Batch> eval_batches){
    isEvaluating = true; //signal to stop calculation of new gradients;

    float acc = 0.0;
    float loss = 0.0;
    Mini_Batch a;
    vector<TF_Tensor*> outputTensors;
    vector<float> metrics(num_output_metrics, 0.0);
    
    //iterate over evaluation batches
    for(int i = 0; i < eval_batches.size(); i++){
        //run evaluation batch
        outputTensors = run_tf_function(eval_batches[i].inputs, eval_batches[i].mini_batch_size, evaluate_input, evaluate_output, evaluate_input_dims, evaluate_output_dims, evaluate_input_dt, evaluate_output_dt);
        for(int j = 0; j < num_output_metrics; j++){
            metrics[j] += static_cast<float*>(TF_TensorData(outputTensors[j]))[0]; //get average of evaluation results
            TF_DeleteTensor(outputTensors[j]); //delete output tensor
        }
    }
    if (eval_batches.size() > 1){
        cout << "\nEVALUATE -  Rank:" << rank;
        for(int i = 0; i < num_output_metrics; i++){
            float avg_metric = metrics[i]/eval_batches.size();
            cout << ", " << metric_names[i] << ":" << avg_metric;
            metrics[i] = 0;
        }
        std::cout << "\n";
    }
    isEvaluating = false;
    pthread_cond_signal(&empty_list_cond); //signal to resume calculation of new gradients;

    return;
}

//prints training metrics obtained during current epoch and runs model evaluation
bool TensAIR::end_of_epoch(){
    bool converged = false;
    if (std::find(std::begin(ranks_to_print), std::end(ranks_to_print), rank) == std::end(ranks_to_print)){ // if this rank does not print (only check convergence)
        epoch++;
        past_metrics.push_back(metrics_epoch_values);
        converged = model_convergence(this->convergence_factor, this->epochs_for_convergence);
        
        for(int i = 0; i < num_output_metrics; i++){
            //float avg_metric = metrics_epoch_values[i]/metrics_epochs_count[i];
            //std::cout << ", "<< metric_names[i] << ":" << avg_metric;
            metrics_epoch_values[i] = 0;
            metrics_epochs_count[i] = 0;
        }

        //std::cout << endl;

        return converged;
    }


    std::cout << std::endl;
    time_t end;
    time(&end);

    
    cout << "Epoch: "  << epoch << "   - Time: " << fixed
                << (end-start);
        cout << " sec " << endl;
    epoch++;
    time(&start);
    
    std::cout<< "Rank:" << rank;
    past_metrics.push_back(metrics_epoch_values);
    converged = model_convergence(this->convergence_factor, this->epochs_for_convergence);

    for(int i = 0; i < num_output_metrics; i++){
        float avg_metric = metrics_epoch_values[i]/metrics_epochs_count[i];
        std::cout << ", "<< metric_names[i] << ":" << avg_metric;
        metrics_epoch_values[i] = 0;
        metrics_epochs_count[i] = 0;
    }

    cout << endl;
    cout << endl;
    
    evaluate(evaluation_batches);
    if(gradientsApplied / epoch_size == this->epochs){
            MPI_Abort(COMM_WORLD, 1);
    }
    return converged;
    //save("");
}

//based on the current model version, prints progress_bar during training
void TensAIR::progress_bar(bool new_epoch){
    int barWidth = 70;

    if (new_epoch){
        std::cout << "[";
        int pos = barWidth;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " <<  epoch_size*(epoch+1) << "  -  " << 100 << " %\r";
        std::cout.flush();
        return;
    }

    float new_progress = (float)gradientsApplied / epoch_size;
    

    //update progress every 5%
    //if((int)(new_progress*100)%5 != 0 || new_progress - 0.05 <= progress){
    if(new_progress - 0.05 < progress){
        return;
    }
    progress = new_progress;
    
    float percentageCompleted = progress - epoch;

    std::cout << "[";
    int pos = barWidth * percentageCompleted;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << gradientsApplied << "  -  " << int(percentageCompleted * 100.0) << " %\r";
    std::cout.flush();
}



/*
* Instantiate input and output Tensors of a specific Tensorflow function (currently available functions: predict, gradient_calc, apply_gradient, save, and evaluate)
* 
* inputs contais the data of the inputs
* mini_batch_size is the mini_batch_size
* input_signature is the signature of the input Tensors
* output_signature is the signature of the output Tensors
* input_dims is the list of dimentions of the input Tensors
* output_dims is the list of dimentions of the output Tensors
* input_dt is the list of types of the input Tensors
* output_dt is the list of types of the output Tensors
*/
vector<TF_Tensor*> TensAIR::run_tf_function(void** inputs, int mini_batch_size, vector<char*> input_signature, char* output_signature, vector<vector<int64_t>> input_dims, vector<vector<int64_t>> output_dims, vector<TF_DataType> input_dt, vector<TF_DataType> output_dt){
   
    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS INPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
    
    vector<TF_Output> inputs_opout;
    vector<TF_Tensor*> inputs_values;
    
    //malloc input dimentions
    int64_t** input_dim = (int64_t**)malloc(sizeof(int64_t*)*input_dims.size());
    for(int i = 0; i < input_dims.size(); i++){
        input_dim[i] = (int64_t*)malloc(sizeof(int64_t)*input_dims[i].size());
    }

    int* target_data;
    int* context_data;
    int* label_data;
    
    //create input tensors
    for(int i = 0; i < input_signature.size(); i++){
        TF_Operation* input_op = TF_GraphOperationByName(graph, input_signature[i]);
        TF_Output input_opout = {input_op};
        
        //calculate tensor size and dimenstions
        size_t nbytes_input = 1;
        for(int j = 0; j < input_dims[i].size(); j++){
            if(input_dims[i][j] < 0){ //check if dimention is based on mini_batch_size 
                //negative dimentions are based on the mini batch and shall be multiplied by the minibatch size
                nbytes_input *= mini_batch_size * input_dims[i][j] * -1;
                input_dim[i][j] = (int64_t)(mini_batch_size * input_dims[i][j] * -1);
            }else{
                nbytes_input *= input_dims[i][j];
                input_dim[i][j] = (int64_t)input_dims[i][j];
            }
        }
        nbytes_input *= TF_DataTypeSize(input_dt[i]);
        int num_dimentions = (int)input_dims[i].size();
        
        TF_Tensor* input_tensor = TF_NewTensor(input_dt[i], input_dim[i], num_dimentions, inputs[i], nbytes_input, &NoOpDeallocator, 0);
        //// As with inputs, check the values for the output operation and output tensor
        //std::cout << "Input: " << TF_OperationNumOutputs(input_op) << "\n";
        //std::cout << "Input info: " << TF_Dim(input_tensor, 0) << "\n";
        
        
        inputs_opout.push_back(input_opout);
        inputs_values.push_back(input_tensor);
        
    }
    
    /////////////////////////////////////////////////////////////////////////////////////////// PROCESS OUTPUT ///////////////////////////////////////////////////////////////////////////////////////////////////
    vector<TF_Output> outputs_opout;
    vector<TF_Tensor*> outputs_values;
    
    int64_t** output_dim = (int64_t**)malloc(sizeof(int64_t*)*output_dims.size());
    for(int i = 0; i < output_dims.size(); i++){
        output_dim[i] = (int64_t*)malloc(sizeof(int64_t)*output_dims[i].size());
    }

    //allocate output tensors
    for(int i = 0; i < output_dims.size(); i++){
        TF_Operation* output_op = TF_GraphOperationByName(graph, output_signature);
        TF_Output output_opout = {output_op};
        output_opout.index=i;             
        
        //calculate tensor size and dimenstions
        size_t num_bytes_out = 1;
        
        for(int j = 0; j < output_dims[i].size(); j++){
            if(output_dims[i][j] < 0){ //check if dimention is based on mini_batch_size
                num_bytes_out *= mini_batch_size * output_dims[i][j] * -1;
                output_dim[i][j] = (int64_t)(mini_batch_size * output_dims[i][j] * -1);
            }else{
                num_bytes_out *= output_dims[i][j];
                output_dim[i][j] = (int64_t)output_dims[i][j];
            }
            
        }
        num_bytes_out *= TF_DataTypeSize(output_dt[i]);
        
        int nSize = sizeof(output_dim)/sizeof(*output_dim);
        TF_Tensor* output_value = TF_AllocateTensor(output_dt[i], output_dim[i], nSize, num_bytes_out);
        
        outputs_opout.push_back(output_opout);
        outputs_values.push_back(output_value);
        
        //// As with inputs, check the values for the output operation and output tensor
        //std::cout << "Output: " << TF_OperationNumOutputs(output_op) << "\n";
        //std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
    }
    
    //////////////////////////////////////////////////////////////////////////// RUN ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    TF_Tensor** output = runSession(inputs_opout, inputs_values, (int)inputs_values.size(), outputs_opout, outputs_values, (int)outputs_values.size());
    
    //////////////////////////////////////////////////////////////////////////// FREE MEMORY //////////////////////////////////////////////////////////////////////////////////////////
    

    for(int i = 0; i < outputs_values.size(); i++){
        TF_DeleteTensor(outputs_values[i]);
        outputs_values[i] = output[i];
    }
    
    for (int i = 0; i < inputs_values.size(); i++){
        TF_DeleteTensor(inputs_values[i]);
    }
    
    free(output);
    
    for(int i = 0; i < input_dims.size(); i++){
        free(input_dim[i]);
    }
    free(input_dim);
    
    for(int i = 0; i < output_dims.size(); i++){
        free(output_dim[i]);
    }
    free(output_dim);
    
    return outputs_values; //return outputs_values to the method that shall deallocate it.
}

/*
* Identify if model has converged or not. Simply check if the loss from current epoch is convergence_factor hogher than last ones and currentLoss < historicalLoss
* 
* convergence_factor is the minimum factor by wich the loss has to vary (to do not consider that the model converged)
* epochs_for_convergence is the number of epochs that shall be compared (always >= 2)
*/
bool TensAIR::model_convergence(float convergence_factor, int epochs_for_convergence){
    bool converged = false;

    //store only epochs_for_convergence last epochs metrics
    while (past_metrics.size() > epochs_for_convergence){
        past_metrics.erase(past_metrics.begin());
    }

    //wait to have epochs_for_convergence epochs
    if (past_metrics.size() == epochs_for_convergence){
        float currentLoss = past_metrics.back()[0];

        bool didNotConverge = false;

        for (int i = 0; i < epochs_for_convergence-1; i++){
            float historicalLoss = past_metrics[i][0];
            if(historicalLoss - currentLoss > convergence_factor*epoch_size && currentLoss < historicalLoss){
                didNotConverge = true;
            }
        }

        if(!didNotConverge){
            converged = true;
        }
    }
    return converged;

}

void TensAIR::copyDataToTensor(TF_Tensor* tensor, void* data){
    void* buffer = TF_TensorData(tensor);
    memcpy(buffer, data, TF_TensorByteSize(tensor));
}


void TensAIR::copyDataToTensors(TF_Tensor** tensors, void** data, int n_tensors){
    for(int i = 0; i < n_tensors; i++){
        copyDataToTensor(tensors[i], data[i]);
    }
}


pair<bool,bool> TensAIR::broadcast_gradient(){
    retrieve_delta();
    
    float** metrics_data = (float**)malloc(sizeof(float*)*num_output_metrics);
    for(int i = 0; i < num_output_metrics; i++){
        metrics_data[i] = static_cast<float*>(TF_TensorData(this->delta_tensors[i]));
    }
    
    pair<bool,bool> afp = after_gradient_application(metrics_data, num_output_metrics, local_gradient_applied);
    
    
    if(worldSize == 1){
        for(int i = 0; i < this->delta_tensors.size(); i++){ //free out_tensors
            TF_DeleteTensor(this->delta_tensors[i]);
        }
        this->delta_tensors.clear();
        this->delta_tensors.shrink_to_fit();
        return afp;
    }
    if(this->delta_tensors.empty()){
        cout << "\n\nError, gradient being broadcasted is empty!\n\n";
        return make_pair(false, true);
    }
    
    //serialize gradients to message2
    message_ptr message_out_tensors = construct_Message_Tensors(this->delta_tensors, num_output_metrics, this->mini_batch_size, local_gradient_applied);
    
    local_gradient_applied = 0;
    
    // add message to output (to send to be applied by other models)
    vector<output_data> res;
    res.push_back(make_pair(std::move(message_out_tensors), dest_broadcast));
    this->send_to_specific_operator(std::move(res), 0); //send gradients to gradient application
    
    for(int i = 0; i < this->delta_tensors.size(); i++){ //free out_tensors
        TF_DeleteTensor(this->delta_tensors[i]);
    }

    clear_delta();
    
    free(metrics_data);
    return afp;
}

void TensAIR::retrieve_delta(){
  
    //retrieve_delta in TF model
    vector<char*> retrieve_delta_outputs(1,retrieve_delta_output);
    pair<vector<TF_Output>, vector<TF_Tensor*>> result = allocateTensor(mini_batch_size, retrieve_delta_outputs, retrieve_delta_output_dims, retrieve_delta_output_dt);
    retrieve_delta_out_op = result.first;
    retrieve_delta_out_tensors = result.second;
    n_retrieve_delta_out = (int)retrieve_delta_output_dims.size();
    
    //parse to pointers instead of vectors
    TF_Output* tar_operators = (TF_Output*) malloc(sizeof(TF_Output)*n_retrieve_delta_out);
    TF_Tensor** tar_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*n_retrieve_delta_out);
    //copy pointers (not necessary to copy the data)
    for (int i = 0; i < n_retrieve_delta_out; i ++){
        tar_operators[i] = retrieve_delta_out_op[i];
        tar_values[i] = retrieve_delta_out_tensors[i];
    }
    
    //retrieve_delta
    TF_SessionRun(session, nullptr,
                retrieve_delta_operators, retrieve_delta_values, n_retrieve_delta_inp,
                tar_operators, tar_values, n_retrieve_delta_out,
                nullptr, 0, nullptr, status);
    
    
    for(int i = 0; i < retrieve_delta_out_tensors.size(); i++){
        TF_DeleteTensor(retrieve_delta_out_tensors[i]);
    }
    
    free(tar_operators);
    
    
    //populate delta (to be used during broadcast)
    bool empty_delta_tensors = delta_tensors.empty();
    for(int i = 0; i < n_retrieve_delta_out; i++){
        if(!empty_delta_tensors){
            TF_DeleteTensor(delta_tensors[i]);
            delta_tensors[i] = tar_values[i];
        }else{
            delta_tensors.push_back(tar_values[i]);
        }
    }
    
    free(tar_values);

    return;
}

void TensAIR::clear_delta(){
    this->delta_tensors.clear();
    this->delta_tensors.shrink_to_fit();

    //clear delta in TF model
    vector<char*> clear_delta_outputs(1,clear_delta_output);
    pair<vector<TF_Output>, vector<TF_Tensor*>> result = allocateTensor(mini_batch_size, clear_delta_outputs, clear_delta_output_dims, clear_delta_output_dt);
    clear_delta_out_op = result.first;
    clear_delta_out_tensors = result.second;
    n_clear_delta_out = (int)clear_delta_output_dims.size();
    
    //parse to pointers instead of vectors
    TF_Output* tar_operators = (TF_Output*) malloc(sizeof(TF_Output)*n_clear_delta_out);
    TF_Tensor** tar_values = (TF_Tensor**) malloc(sizeof(TF_Tensor*)*n_clear_delta_out);
    //copy pointers (not necessary to copy the data)
    for (int i = 0; i < n_clear_delta_out; i ++){
        tar_operators[i] = clear_delta_out_op[i];
        tar_values[i] = clear_delta_out_tensors[i];
    }
    
    //clear delta
    TF_SessionRun(session, nullptr,
                clear_delta_operators, clear_delta_values, n_clear_delta_inp,
                tar_operators, tar_values, n_clear_delta_out,
                nullptr, 0, nullptr, status);
    
    
    for(int i = 0; i < clear_delta_out_tensors.size(); i++){
        TF_DeleteTensor(clear_delta_out_tensors[i]);
        TF_DeleteTensor(tar_values[i]);
    }
    
    free(tar_operators);
    
    return;
}

int TensAIR::print_to_file_training(float** metrics_data, int n_metrics, int n_delta){
    // Check if the file is open
    if (!this->file_to_print.is_open()) {
        std::cerr << "Error opening file to print to!" << std::endl;
        return 1;
    }

    file_to_print << "training, " << gradientsApplied << ",";
    
    //add metrics to file in csv format
    for(int i = 0; i < n_metrics; i++){
        if(i+1 == n_metrics){ // if last metric, end line
            file_to_print << metrics_data[i][0]/n_delta << std::endl;
        }else{
            file_to_print << metrics_data[i][0]/n_delta << ",";
        }
    }
    return 0;
}


pair<bool,bool> TensAIR::after_gradient_application(float** metrics_data, int n_metrics, int n_delta){
    int former_update = gradientsApplied;
    gradientsApplied += n_delta;
    
    // check if results should be printed to file
    if (floor(former_update / print_frequency)  < floor(gradientsApplied / print_frequency)){
        
        if(print_to_file_training(metrics_data, n_metrics, n_delta)){
            return make_pair(false, true);
        }
    }
    
    update_metrics(num_output_metrics, metrics_data, n_delta);
    
   
    unsigned long int new_model_version = this->models_version[this->rank];
    unsigned long int old_model_version = this->models_version[this->rank] - broadcast_frequency;
    
    bool converged = false;

    //if end of epoch and current rank is in "ranks_to_print", print progress_bar
    if (floor(former_update / epoch_size)  < floor(gradientsApplied / epoch_size)){
        progress_bar(true);
        converged = end_of_epoch();
    }

    //if current rank is in "ranks_to_print", print progress_bar
    if(std::find(std::begin(ranks_to_print), std::end(ranks_to_print), rank) != std::end(ranks_to_print)){
        progress_bar(false);
    }
    
    bool end_training_bool = end_training(metrics_data, n_metrics, n_delta);
    

    return make_pair(converged, end_training_bool);
}


bool TensAIR::end_training(float** metrics_data, int n_metrics, int n_delta){
    if(epoch == this->epochs){
            return true;
    }
    return false;
}