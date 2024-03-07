# TensAIR
<div style="text-align: justify;">
  
**TensAIR** is a distributed framework for **training and predicting in ANNs models in real-time**. TensAIR extends the [AIR](https://gitlab.uni.lu/mtheobald/AIR) stream-processing engine, which allows **asynchornous and descentralized** processing of its dataflow operators (supporting the common dataflow operators like *Map*, *Reduce*, *Join*, etc.) in addition to new **train_step** and **predict** Onlie Learning (OL) functions. TensAIR implements the **TensorFlow C API** library on top of **AIR**, which means that it can scale out both the training and prediction tasks of an ANN model to multiple compute nodes, either with or without **GPUs** associated with them.

Additionally, TensAIR supports both passive and active concept drift adaptation strategies. For active drift adaptation, one may instanciate the DriftDetector operator, which implements the [**OPTWIN**](https://github.com/maurodlt/OPTWIN) concept drift detector. For passive drift adaptation, one shall just set the variable *drift_detector_mode = TensAIR::Drift_Mode::AUTOMATIC*.

</div>

Build & Run:
------------
### Install Dependencies:

- MPI; preferably OpenMPI/MPICH2 

- cmake

- pybind11 

- mpi4py 

- TensorFlow C API; version  above 2.3 and below 2.9.1 ([tutorial]([https://www.tensorflow.org/install](https://www.tensorflow.org/install/lang_c)))

- [saved_model_cli](https://www.tensorflow.org/guide/saved_model#install_the_savedmodel_cli) (automatically installed with TensorFlow)

**Note**: saved_model_cli shall be on PATH when running TensAIR (usually automatic after installing TensorFlow)

  On Mac(intel): 
  ```sh
    # Instal MPI
    brew install openmpi
    # Instal CMake
    brew install cmake

    # Install TensorFlow C API
    wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-2.9.1.tar.gz
    mkdir tensorflow_c_api
    tar -C tensorflow_c_api -xzf libtensorflow-cpu-darwin-x86_64-2.5.0.tar.gz
    echo "export LIBRARY_PATH=$LIBRARY_PATH:$PWD/tensorflow_c_api/lib" >> ~/.bashrc
    export LIBRARY_PATH=$LIBRARY_PATH:"$PWD/tensorflow_c_api/lib"
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:"$PWD/tensorflow_c_api/lib"

    #install pybind, mpi4py, and tensorflow
    pip install pybind11 mpi4py tensorflow==-2.9.1
  ```
  On Mac(arm m1/m2): 
  ```sh
    # Instal MPI
    brew install openmpi
    # Instal CMake
    brew install cmake

    # Install TensorFlow C API for Linux arm (workaround to do not have to build it from source) (https://gist.github.com/wangjia184/f9ffb2782d0703ef3dbceec9b2bbc4b4?permalink_comment_id=4269188#gistcomment-4269188)
    brew install libtensorflow
    cd $(brew --prefix libtensorflow)/lib #go to location in which libtensorflow was installed
    # Create links of the libraries with names on mac standard
    ln -s libtensorflow.TF_VERSION.dylib libtensorflow.so.TF_VERSION #E.g. ln -s libtensorflow.2.9.1.dylib libtensorflow.so.2.9.1
    ln -s libtensorflow.TF_VERSION.dylib libtensorflow.so.TF_VERSION #E.g. ln -s libtensorflow.2.dylib libtensorflow.so.2
    ln -s libtensorflow.dylib libtensorflow.so
    echo "export LIBRARY_PATH=$LIBRARY_PATH:$PWD/tensorflow_c_api/lib" >> ~/.bashrc
    export LIBRARY_PATH=$LIBRARY_PATH:"$PWD/tensorflow_c_api/lib"
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:"$PWD/tensorflow_c_api/lib"
    
    #install pybind and mpi4py, and tensorflow
    pip install pybind11 mpi4py tensorflow_macos==2.9.1
  ```

  On Linux: 
  ```sh
    # Install MPI
    sudo apt-get install mpich
    # Install MPI
    sudo apt-get install cmake
  
    # Instal TensorFlow C API
    wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.9.1.tar.gz
    mkdir tensorflow_c_api
    tar -C tensorflow_c_api -xzf libtensorflow-cpu-linux-x86_64-2.9.1.tar.gz
    echo "export LIBRARY_PATH=$LIBRARY_PATH:$PWD/tensorflow_c_api/lib" >> ~/.bashrc
    export LIBRARY_PATH=$LIBRARY_PATH:"$PWD/tensorflow_c_api/lib"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$PWD/tensorflow_c_api/lib"
    

    #install pybind and mpi4py
    pip install pybind11 mpi4py tensorflow==-2.9.1
  ```
### Build and Run TensAIR

- Download the project
```sh
  git clone https://github.com/maurodlt/TensAIR 
```

- Build the project
```sh
  cd TensAIR
  echo "export TENSAIR_PATH=$PWD" >> ~/.bashrc #Add TensAIR directory to the path at every new session. 
  source configure.sh  #Add paths to pre-installed libraries (they are usually automatically recognized).
  mkdir Release
  cd Release
  cmake ../
  make all -j$(nproc)
  
```

- Run a use-case
  - **Important**: Current models in this repository were generated using Tensorflow 2.9.1. Therefore, if using a different Tensorflow version, it may be necessary to regenerate the models using the following notebooks: [W2V-Model.ipynb](https://github.com/maurodlt/TensAIR/tree/main/Examples/W2V/W2V-Model.ipynb), [CIFAR-Model.ipynb](https://github.com/maurodlt/TensAIR/tree/main/Examples/CIFAR/CIFAR-Model.ipynb), and [DEMO-Model.ipynb](https://github.com/maurodlt/TensAIR/tree/main/Examples/DEMO/DEMO-Model.ipynb)
```sh
  mpirun -np <no.of dataflows> ../lib/TensAIR <use-case abbreviation>
  #(E.g., mpirun -np 2 ./TensAIR DEMO)
  #(E.g., mpirun -np 8 ./TensAIR CIFAR)
  #(E.g., mpirun -np 4 ./TensAIR W2V) <-Previous creation of model and dataset necessary (just run /Examples/W2V/W2V-Model.ipynb and /Examples/W2V/W2V_data.ipynb)
  ```  

- Run using Python Interface
  - Create python script with dataflow.

  ```py
  from mpi4py import MPI
  import tensair_py
  import os
  comm = MPI.COMM_WORLD
  
  tensair_path = os.environ.get("TENSAIR_PATH") #retrieve TENSAIR_PATH

  mini_batch_size = 128 #set desired mini-batch-size
  msg_sec = 100  #set throughput of data stream (number of minibatches generated per second)  
  init_code= tensair_path + "/Examples/demo/Demo_Init.py" #Python file implementing next_message method that receives mini_batch_size as input and returns serialized minibatch

  ######## Instanciate Event Generator
  event_generator = tensair_py.UDF_EventGenerator(mpi_comm=comm, tag=1, mini_batch_size=mini_batch_size, msg_sec=msg_sec, init_code=init_code)
      
  #Define maximum message size received by TensAIR model
  inputMaxSize = 4+4+ (8*2) + (4*mini_batch_size) + (4*mini_batch_size*32*32*3) # Minibatch size + metadata
  gradientsMaxSize = 8+4+4+4+(8*6) + 4 + 4 + (8*((64*10) + (10) + (1024*64) + (64))) #Gradients size + metadata
  window_size = max(inputMaxSize, gradientsMaxSize)

  ######## Instanciate TensAIR
  saved_model_dir = tensair_path + "/data/demo/cifar_model_demo.tf" #TF Model directory (created using /Examples/DEMO/DEMO-Model.ipynb)
  model = tensair_py.TensAIR(mpi_comm=comm, tag=2, window_size=window_size, saved_model_dir=saved_model_dir)
  
  ######## Instanciate Drift Detector (OPTWIN)
  drift_detector = tensair_py.OPTWIN_drift_detector(mpi_comm=comm, tag=3)

  operators = [event_generator, model, drift_detector] #list operators
  links = [[0,1],[1,1],[1,2],[2,1]] #link operators

  ######## Instanciate Dataflow
  basic_dataflow = tensair_py.BasicDataflow(mpi_comm=comm, operators=operators, links=links)
  print("Starting dataflow")
  basic_dataflow.streamProcess() #start dataflow
  ```

  - Run python script
    ```sh
    mpirun -np <no.of dataflows> python <script_location>
    #(E.g. mpirun -np 2 python $TENSAIR_PATH/Examples/DEMO/DEMO-Run.py)
    #(E.g. mpirun -np 2 python $TENSAIR_PATH/Examples/CIFAR/CIFAR-Run.py)
    #(E.g. mpirun -np 2 python $TENSAIR_PATH/Examples/W2V/W2V-Run.py) <-Previous creation of model and dataset necessary (just run /Examples/W2V/W2V-Model.ipynb and /Examples/W2V/W2V_data.ipynb
    ```

- Run using SLURM
```sh
  srun -n <no.of dataflows> $TENSAIR_PATH/lib/TensAIR <use-case abbreviation>
```

Run under *OVERCOMMIT* option
```sh
  srun --overcommit -n <no.of dataflows> $TENSAIR_PATH/lib/TensAIR <use-case abbreviation>
```

Results visualization:
------------
  - The results are stored in the /output directory in .csv with the following format:
    - In case of prediction:  
      - predicting, iteration, loss, prediction
    - In case of training:
      - training, iteration, loss, accuracy
     
  - To visualize TensAIR loss over time one may simply plot it using matplotlib as exemplified in  the following notebooks: [CIFAR-Run.ipynb](https://github.com/maurodlt/TensAIR/blob/main/Examples/CIFAR/CIFAR-Run.ipynb), [W2V-Run.ipynb](https://github.com/maurodlt/TensAIR/blob/main/Examples/W2V/W2V-Run.ipynb), and [DEMO-Run.ipynb](https://github.com/maurodlt/TensAIR/blob/main/Examples/DEMO/DEMO-Run.ipynb). Below, there is an example such plot.
    ![Plot of DEMO usecase](demo_usecase_plot.svg)

Available usecases:
------------
 - Word2Vec (W2V)
 - CIFAR-10 (CIFAR)
 - CIFAR-10 with active drift adaptation (DEMO)

Cite us:
------------

Mauro D. L. Tosi, Vinu E. Venugopal, and Martin Theobald. 2024. TensAIR: Real-Time Training of Neural Networks from Data-streams. In 2024 The 8th International Conference on Machine Learning and Soft Computing (ICMLSC 2024), January 26--28, 2024, Singapore, Singapore. ACM, New York, NY, USA 10 Pages. https://doi.org/10.1145/3647750.3647762
