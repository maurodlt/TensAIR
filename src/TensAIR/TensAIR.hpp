#ifndef TENSAIR_DEF
#define TENSAIR_DEF


#ifndef SAVED_MODEL_CLI_PATH
#define SAVED_MODEL_CLI_PATH "" //hard code saved_model_cli if necessary for debugging in XCode
#endif

#define OUTPUT_METRICS 2 //number of output metrics, currently only Loss and Accuracy are supported


#pragma once
#include "../dataflow/BasicVertex.hpp"
#include <vector>
#include <tensorflow/c/c_api.h>
#include <fstream>

/**
 * Mini_batch received by TensAIR from EventGenerator
 * 
 * num_output_metrics currently is fixed to 2 metrics (Loss and Accuracy) 
 * size_inputs is a list containing the sizes of each input
 * inputs is a list of num_inputs inputs of sizes size_inputs
 */ 
struct Mini_Batch {
    int mini_batch_size;
    int num_output_metrics=OUTPUT_METRICS;
    int num_inputs;
    size_t* size_inputs;
    void** inputs;
};
typedef struct Mini_Batch mini_batch;


/**
 * Tensors received by TensAIR from TensAIR containing Tensors(gradients) to be applied locally
 * 
 * model_version is the number of gradients applied to the model that calculated this gradient
 * num_output_metrics currently is fixed to 2 metrics (Loss and Accuracy) 
 * size_metrics is a list containing the sizes of each metric
 * size_gradients is a list containing the sizes of each Tensor
 * metrics_data is a list of metrics (metrics shall be float)
 * gradients_data is a list of Tensors containing the gradients.
 */ 
struct Tensor_Data {
    int model_version;
    int mini_batch_size;
    int n_delta;
    int num_output_metrics=2;
    int num_gradients;
    size_t* size_metrics;
    size_t* size_gradients;
    float** metrics_data;
    void** gradients_data;
};
typedef struct Tensor_Data tensor_sata;


///TensAIR performs all TensorFlow operations in AIR
class TensAIR : public BasicVertex<>{
    
    public:
    
        // Beaviour of TensAIR in relation of the concept drift adaptation
        enum class Drift_Mode {
            AUTOMATIC, //active drift adaptation
            ALWAYS_TRAIN, //passive drift adaptation
            NEVER_TRAIN //no drift adaptation
        };


        /**
         * Default Constructor
         * 
         * tag is the number of this operator in the AIR dataflow
         * rank is the number of this rank on MPI
         * worldSize is the total number of ranks on MPI
         * windowSize is the maximum message size received by TensAIR (either from the EventGenerator or TensAIR itself) <- max(Mini_Batch, Tensor_Data)
         * broadcast_frequency is the number of gradients that shall be calculated before broadcasting them to other ranks
         * epochs is the total number of epochs before stopping dataflow
         * gpus_per_node is the total number of GPUs that are available per node
         * saved_model_dir is the model.tf directory
         * eval_data_file is a binary file containing data to be used for evaluation of the model (formatted with the Mini_Batch struct format, the same received by the Python interface)
         * tags is the tensorflow tag used to run the TensorFlow methods (usually "serve")
         * epoch_size is the number of mini_batches that indicate 1 epoch
         * convergence_factor is the minimum factor by which the loss of the current epoch have to vary in relation to past (epochs_for_convergence) epochs to do not consider that the model converged
         * epochs_for_convergence is the number of epochs that will be compared with the current epoch to determine convergence
         * drift_detector_mode is set to determine if the concept drift detector will be used, if the model will always train, or if it will never be trained
         * print_to_folder is the folder in which reuslts will be printed
         * print_frequency is the number of gradients that shall be applied before updating the results file
         * preallocate_tensors is to determine if the input tensors shall be pre-allocatted, it speeds up the training but the mini_batch_size shall be fixed and determined beforehand
         * mini_batch_size is the mini_batch_size value that shall be used during the whole training and prediction (keep 0 if this number is unknow or may vary during the stream)
         * comm is the MPI object received when using the Python Interface
         * 
         * IMPORTANT: the constructor calls load_tensors(). This method cannot identify the dimention of tensors that vary based on a variable other than mini_batch size. If this is the case, manually specify the tensor dimention after initializing the TensAIR object (before training/evaluating)
         */
        TensAIR(const int tag, const int rank, const int worldSize, int windowSize, int broadcast_frequency, int epochs, int gpus_per_node, const char* saved_model_dir, const char* eval_data_file, const char* tags, int epoch_size = 1000, float convergence_factor = 1e-2, int epochs_for_convergence=2, Drift_Mode drift_detector_mode = TensAIR::Drift_Mode::AUTOMATIC, std::string print_to_folder = "", int print_frequency = 10, bool preallocate_tensors = false, int mini_batch_size = 0, MPI_Comm comm = MPI_COMM_WORLD);
        
        ///Default destructor
        virtual ~TensAIR();

        /**Main method that manages TensAIR dataflow**/
        void streamProcess(int channel);
        

        int num_output_metrics = OUTPUT_METRICS; ///number of metrics outputted by model (CURRENTLY AWAYS 2)
        std::vector<char const*> metric_names = {"Loss", "Accuracy"}; ///list of name of the metrics outputted by model (CURRENTLY AWAYS ACCURACY AND LOSS)
        
        //predict
        std::vector<char*> predict_input; ///list of input tensor names for prediction
        char* predict_output; ///output tensor name for prediction
        std::vector<std::vector<int64_t>> predict_input_dims; ///dimentions of input tensors for prediction 
        std::vector<std::vector<int64_t>> predict_output_dims; ///dimentions of putput tensors for prediction
        std::vector<TF_DataType> predict_input_dt; ///type of input tensors for prediction
        std::vector<TF_DataType> predict_output_dt; ///type of putput tensors for prediction
        
        //apply_gradient
        std::vector<char*> apply_gradient_input; ///list of input tensor names for apply_gradient 
        char* apply_gradient_output; ///output tensor name for apply_gradient
        std::vector<std::vector<int64_t>> apply_gradient_input_dims; ///dimentions of input tensors for apply_gradient 
        std::vector<std::vector<int64_t>> apply_gradient_output_dims; ///dimentions of putput tensors for apply_gradient
        std::vector<TF_DataType> apply_gradient_input_dt; ///type of input tensors for apply_gradient
        std::vector<TF_DataType> apply_gradient_output_dt; ///type of putput tensors for apply_gradient

        //save
        std::vector<char*> save_input; ///list of input tensor names for save 
        char* save_output; ///output tensor name for save
        std::vector<std::vector<int64_t>> save_input_dims; ///dimentions of input tensors for save 
        std::vector<std::vector<int64_t>> save_output_dims; ///dimentions of putput tensors for save
        std::vector<TF_DataType> save_input_dt; ///type of input tensors for save
        std::vector<TF_DataType> save_output_dt; ///type of putput tensors for save
        
        //evaluate
        std::vector<char*> evaluate_input; ///list of input tensor names for evaluate 
        char* evaluate_output; ///output tensor name for evaluate
        std::vector<std::vector<int64_t>> evaluate_input_dims; ///dimentions of input tensors for evaluate 
        std::vector<std::vector<int64_t>> evaluate_output_dims; ///dimentions of putput tensors for evaluate
        std::vector<TF_DataType> evaluate_input_dt; ///type of input tensors for evaluate
        std::vector<TF_DataType> evaluate_output_dt; ///type of putput tensors for evaluate

        //clear_delta
        std::vector<char*> clear_delta_input; ///list of input tensor names for clear_delta
        char* clear_delta_output; ///output tensor name for clear_delta
        std::vector<std::vector<int64_t>> clear_delta_input_dims; ///dimentions of input tensors for clear_delta
        std::vector<std::vector<int64_t>> clear_delta_output_dims; ///dimentions of putput tensors for clear_delta
        std::vector<TF_DataType> clear_delta_input_dt; ///type of input tensors for clear_delta
        std::vector<TF_DataType> clear_delta_output_dt; ///type of putput tensors for clear_delta
    
        //train_step
        std::vector<char*> train_step_input; ///list of input tensor names for train_step
        char* train_step_output; ///output tensor name for train_step
        std::vector<std::vector<int64_t>> train_step_input_dims; ///dimentions of input tensors for train_step
        std::vector<std::vector<int64_t>> train_step_output_dims; ///dimentions of putput tensors for train_step
        std::vector<TF_DataType> train_step_input_dt; ///type of input tensors for train_step
        std::vector<TF_DataType> train_step_output_dt; ///type of putput tensors for train_step
        
        //retrieve_delta
        std::vector<char*> retrieve_delta_input; ///list of input tensor names for retrieve_delta
        char* retrieve_delta_output; ///output tensor name for retrieve_delta
        std::vector<std::vector<int64_t>> retrieve_delta_input_dims; ///dimentions of input tensors for retrieve_delta
        std::vector<std::vector<int64_t>> retrieve_delta_output_dims; ///dimentions of putput tensors for retrieve_delta
        std::vector<TF_DataType> retrieve_delta_input_dt; ///type of input tensors for retrieve_delta
        std::vector<TF_DataType> retrieve_delta_output_dt; ///type of putput tensors for retrieve_delta

    protected:
            
            virtual bool message_from_generator(message_ptr message); //process message received from the mini_batch_generator
            virtual bool message_from_model(message_ptr message); //process message received from other models (gradients or convergence flag)
            virtual void message_from_drift_detector(message_ptr message); //process message received from drift detector (concept drift flag)

            /**
             * Sends multiple messages to one of the next (specific) operators in the dataflow.
             * It is possible to modify which ranks will receive the messages
             * by setting up correctly the target_ranks vector.
             * 
             * It is also possible to define which is the target operator. targetOperator_index = position of the operator in the links defined in the dataflow (in respect to current operator)
             * */
            void send_to_specific_operator(vector<output_data> messages, int targetOperator_index);

            static void NoOpDeallocator(void* data, size_t a, void* b) {} ///Used by TF when allocating memory for Tensors

            /*
            * Method that loads metadata information from input and output tensor
            * 
            * tensors trackes:
            *   - apply_grad
            *   - evaluate
            *   - save
            *   - prediction
            *   - clear_delta
            *   - train_step
            *   - retrieve_delta
            * Note: If the tensors dimentions depend on external variables (other than mini_batch_size), they cannot be directly identified using saved_model_cli. Then, it is necessary to override this and define them manually
            */
            void load_tensors(const char* saved_model_dir, const char* tags);

            // preallocate input tensors
            void pre_allocate_tensors(int mini_batch_size);

            // preallocate input tensors that do not depend on the mini_batch_size
            void pre_allocate_base_tensors();

            // allocate single tensor
            pair<vector<TF_Output>, vector<TF_Tensor*>> allocateTensor(int mini_batch_size, vector<char*> signature, vector<vector<int64_t>> dims, vector<TF_DataType> dt);

            // execute shell command from input and returns the shell output as string
            static std::string exec_shell_command(const char* cmd, int try_times, int max_try_times = 5); 
            
            /*
            * Static method that fills input and output tensor details 
            * 
            * parses saved_model_cli string output and store parsed result in the pointers received as input.
            * Note: all outputs have the same name, which the method returns as a string and must be set (on the respective class variable) by the method that receives it.
            */
            static string process_tensor_cli(string cli_output, vector<char*> &tensors, vector<vector<int64_t>> &tensor_dim, vector<TF_DataType> &tensor_dt); 


            /*
            * construct Mini_Batch struct from message arriving from EventGenerator
            *
            *   MESSAGE FORMAT
            *   mini_batch_size                                                         (int mini_batch_size)
            *   num_inputs                                                              (int num_inputs)
            *   size_input_0, size_inpu_1, ... , size_input_num_inputs                  (int[num_inputs] size_inputs) [note: size in bytes]
            *   input_0, input_1, ... , input_num_inputs                                (char[num_inputs][size_input[num_input]])
            */
            Mini_Batch read_MiniBatch(message_ptr message); 

            //receives message from Event Generator and returns message with Tensors(gradients) (to send to all TensAIR ranks) & loss (to send to drift detector)
            virtual void train_step(message_ptr message);

            //receives message from Event Generator and returns message with predictions (to send to next Vertex on dataflow) & loss (to send to drift detector)
            vector<output_data> predict(message_ptr message);

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
            message_ptr construct_Message_Tensors(vector<TF_Tensor*> out_tensors, int num_output_metrics, int mini_batch_size, int n_delta);

            /*
            * serializes Tensors to a message, which will be send to drift detector
            * MESSAGE2 FORMAT (to send to drift detector)
            * loss                                                                    (int)
            * reset_detector                                                          (int) 
            */
            message_ptr construct_Message_Tensors_Loss(vector<TF_Tensor*> out_tensors, int num_output_metrics);

            /*
            * Fetches new messages from the input buffer.
            *     Updates in the model are inserted at the beggining of the list. Then, the train steps are inserted at the end of it.
            */
            virtual void fetchUpdateMessages(int channel, list<message_ptr>& lis);
            
            //fetch messages from lis buffer (which arrived from other TensAIR ranks) 
            virtual message_ptr fetchNextMessage(int channel, list<message_ptr>& pthread_waiting_list); 
            

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
            Tensor_Data read_Tensors(message_ptr message);

            //updates local metrics based on metrics from new gradients (METRICS MUST BE FLOAT)
            void update_metrics(int num_output_metrics, float** metrics_data, int n_delta); ///updates local metrics based on metrics from new gradients
            
            // receives message from other TensAIR rank and apply it locally.
            virtual pair<bool,bool> apply_gradient(message_ptr message);

            //prints progress_bar during training
            void progress_bar(bool new_epoch); 
            
            //prints training metrics obtained during current epoch and runs model evaluation
            bool end_of_epoch(); 
            
            /*
            * Identify if model has converged or not. Simply check if the loss from current epoch is convergence_factor hogher than last ones and currentLoss < historicalLoss
            * 
            * convergence_factor is the minimum factor by wich the loss has to vary (to do not consider that the model converged)
            * epochs_for_convergence is the number of epochs that shall be compared (always >= 2)
            */
            bool model_convergence(float convergence_factor = 1e-3, int epochs_for_convergence = 2);
        
            /*
            * Reads evaluation datatset (divided in batches) from a file
            * 
            * The file should be written on the following format:
            * mini_batch_size num_training_examples num_tensors size_tensor_0 tensor_0 ... size_tensor_n tensor_n
            * num_training_examples, num_tensors, size_tensor_i (in bytes): int
            * tensor_i : size_input_i * bytes
            * IMPORTANT: char** tensors is never deallocated since it will be used for evaluating the model indefinitely!
            */
            vector<Mini_Batch> loadEvaluationBatches(const char* file);

            // evaluates model using evaluation data received in the constructor
            void evaluate(vector<Mini_Batch> eval_batches); 

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
            TF_Tensor** runSession(vector<TF_Output> input_operator, vector<TF_Tensor*> input_value, int nInputs, vector<TF_Output> target_operator, vector<TF_Tensor*> target_value, int nOutputs);

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
            vector<TF_Tensor*> run_tf_function(void** inputs, int mini_batch_size, vector<char*> input_signature, char* output_signature, vector<vector<int64_t>> input_dims, vector<vector<int64_t>> output_dims, vector<TF_DataType> input_dt, vector<TF_DataType> output_dt);

            
            void copyDataToTensor(TF_Tensor* tensor, void* data); //copy data to tensor
            void copyDataToTensors(TF_Tensor** tensors, void** data, int n_tensors); //copy data to multiple tensors
            pair<bool,bool> broadcast_gradient(); //broadcast gradient to other models
            pair<bool,bool> after_gradient_application(float** metrics_data, int n_metrics, int n_delta); //update metrics, verify convergence, print reuslts 
            void clear_delta(); //clear delta in TF model
            void retrieve_delta(); //retrieve delta from TF model
            
            TF_Graph* graph; ///Model TF graph
            TF_Status* status; ///Model TF status
            TF_SessionOptions* SessionOpts; ///Model TF options
            TF_Buffer* RunOpts; ///Model used by TensorFlow when loading the model
            TF_Session* session; ///Model TF session
            list<message_ptr> model_update_list; ///buffer of messages received from other TensAIR rank
            list<message_ptr> drift_list; ///buffer of messages received from Drift Detector
            pthread_mutex_t update_list_mutex = PTHREAD_MUTEX_INITIALIZER; ///mutex that identifies if the incoming messages buffer is being accessed
            pthread_cond_t update_list_cond = PTHREAD_COND_INITIALIZER; ///signal to mutex that identifies if the incoming messages buffer is being accessed
            pthread_mutex_t empty_list_mutex = PTHREAD_MUTEX_INITIALIZER; ///mutex that identifies if evaluation is currently in course (cannot train during evaluation)
            pthread_cond_t empty_list_cond = PTHREAD_COND_INITIALIZER; ///signale mutex that identifies if evaluation is currently in course (cannot train during evaluation)
            int model_update_rank = this->worldSize; ///channel of rank that will perform updates in local model
            int model_update_drift_rank = 2*this->worldSize; ///channel of rank that will perform updates drift status 
            int broadcast_frequency; ///broadcast_frequency is the number of gradients that shall be calculated before broadcasting them with the other ranks (UNDER DEVELOPMENT, USE 1)
            int epochs; ///number of epochs ran before stopping dataflow
            int local_gradients_count = 0; ///number of updates performed on local model
            int gpus_per_node = 0; ///number of gpus available per node
            bool isEvaluating = false; ///flag that indicates if model is currently under evaluation (do not calculate new gradients in the meantime)
            vector<vector<float>> past_metrics; ///stores the avg of metrcis on previous epochs
            unsigned long int *models_version; ///model version of all models
            int epoch = 0; ///curent epoch
            float progress = -0.05; ///step size for progress_bar
            vector<int> ranks_to_print = {0}; ///list of ranks that will have their progress_bar printed
            int epoch_size; ///number of mini_batches that indicate 1 epoch
            vector<Mini_Batch> evaluation_batches; ///store evaluation batches used on loadEvaluationBatches, which is called in the constructor
            int drift_magnetude = 0.0; //how strong is the concept drift (how much should we adapt the learning rate)
            int drift_identifier_ranks = 0; //list of ranks that will detect concept drift
            TensAIR::Drift_Mode drift_detector_mode = TensAIR::Drift_Mode::AUTOMATIC; // 1=automatically drift detection&adaptation, 2=always train, 3=never train
            long long int count_predictions = 0; //number of predictions performed
            std::ofstream file_to_print; //file to print results
            std::string print_to_file = ""; //name of file to print results
            int print_frequency; //frequency results will be printed
            bool has_converged = false; //determine if model has converged or needs to be trained
            float convergence_factor; //minimum amount avg loss has to vary to determine model has not converged yet
            int epochs_for_convergence; //number of epochs that shall be analyzed to determine convergence
            bool reset_drtift_detector = false; //does the drift detector needs to be reset in the next iteration?
            bool preallocate_tensors; //shall TensAIR pre-allocate tensors based on a fixed mini_batch_size?
            int mini_batch_size; //fixed mini_batch_size (0 if not fixed or unknow)
            int gradientsCalculated = 0; // number of gradients locally calculated
            vector<TF_Tensor*> delta_tensors; // delta of tensors, which shall be broadcasted to other models
            int gradientsApplied = 0; //number of gradients locally applied
            destination dest_broadcast; // destination of broadcasts (every rank besides the one broadcasting)
            int local_gradient_applied = 0; //number of gradients locally applied after the last broadcast
            vector<float> metrics_epoch_values; //sum of past metrics
            vector<int> metrics_epochs_count; //number of metrics summed
            time_t start;

            //apply gradient
            vector<TF_Output> app_inp_op;
            vector<TF_Tensor*> app_inp_tensors;
            int n_app_inp;
            vector<TF_Output> app_out_op;
            vector<TF_Tensor*> app_out_tensors;
            int n_app_out;
            TF_Output* inp_app_operators;
            TF_Tensor** inp_app_values;
            
            //clear delta
            vector<TF_Output> clear_delta_inp_op;
            vector<TF_Tensor*> clear_delta_inp_tensors;
            int n_clear_delta_inp;
            vector<TF_Output> clear_delta_out_op;
            vector<TF_Tensor*> clear_delta_out_tensors;
            int n_clear_delta_out;
            TF_Output* clear_delta_operators;
            TF_Tensor** clear_delta_values;
        
            //train_step
            vector<TF_Output> train_step_inp_op;
            vector<TF_Tensor*> train_step_inp_tensors;
            int n_train_step_inp;
            vector<TF_Output> train_step_out_op;
            vector<TF_Tensor*> train_step_out_tensors;
            int n_train_step_out;
            TF_Output* train_step_operators;
            TF_Tensor** train_step_values;
            
            //retrieve_delta
            vector<TF_Output> retrieve_delta_inp_op;
            vector<TF_Tensor*> retrieve_delta_inp_tensors;
            int n_retrieve_delta_inp;
            vector<TF_Output> retrieve_delta_out_op;
            vector<TF_Tensor*> retrieve_delta_out_tensors;
            int n_retrieve_delta_out;
            TF_Output* retrieve_delta_operators;
            TF_Tensor** retrieve_delta_values;

            //predict
            vector<TF_Output> predict_inp_op;
            vector<TF_Tensor*> predict_inp_tensors;
            int n_predict_inp;
            vector<TF_Output> predict_out_op;
            vector<TF_Tensor*> predict_out_tensors;
            int n_predict_out;
            TF_Output* predict_operators;
            TF_Tensor** predict_values;
};




#endif /*TENSAIR_DEF*/
