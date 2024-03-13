#pragma once
#include "../TensAIR/TensAIR.hpp"
#include <vector>
#include <queue> 
#include <chrono>

struct Mini_Batch_Generator {
    int mini_batch_size;
    int num_inputs;
    size_t *size_inputs;
    char **inputs;
};

struct Dataset {
    int num_inputs;
    vector<pair<char*, char*>> imgs_labels;
};

class VGG16_Convergence : public TensAIR{

    public:

        VGG16_Convergence(const int tag, const int rank, const int worldSize, int windowSize, int broadcast_frequency, int epochs, int gpus_per_node, const char* saved_model_dir, const char* eval_data_file, const char* tags, int epoch_size = 1000, float convergence_factor = 1e-2, int epochs_for_convergence=2, TensAIR::Drift_Mode drift_detector_mode = TensAIR::Drift_Mode::AUTOMATIC, std::string print_to_folder = "", int print_frequency = 10, bool preallocate_tensors = false, int mini_batch_size = 128, MPI_Comm comm = MPI_COMM_WORLD);

        void streamProcess(int channel);
        Dataset readDataset();
        void shuffleDataset();
        std::vector<Mini_Batch_Generator> createMinibatches(Dataset dataset, int mini_batch_size);
        void addToMiniBatch(Mini_Batch_Generator *ubatch, int position);
        void refillMiniBatches();
        Mini_Batch_Generator nextElement();
        message_ptr generateMessage();
        int readTrainingSample(std::ifstream &infile, char* image, char* label_from_image);
        int print_to_file_training(float** metrics_data, int n_metrics, int n_delta);
        bool end_training(float** metrics_data, int n_metrics, int n_delta);

        list<message_ptr> update_list;
        int itr_per_epoch;
        bool warmup = true;
        float decay = 1e-5;
        Dataset dataset;
        const char* train_data_file = "/Users/mauro.dalleluccatosi/Documents/GitHub/tensair-dev/data/TinyImageNet/train.bin"; //file with trining data
        std::vector<Mini_Batch_Generator> data;
        std::chrono::time_point<std::chrono::high_resolution_clock> lastUpdate;
        float currentError = 100000; //high value that will be overwritten
        std::vector<Mini_Batch_Generator>::iterator it;
        std::vector<pair<char*,char*>>::iterator iterator_images_labels;
        std::queue<float> recent_losses;
        float sum_recent_losses;
        int n_recent_losses = 50;
        float loss_objective = 1;
};