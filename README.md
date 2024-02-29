# TensAIR

**TensAIR** is a **distributed stream-processing engine** (supporting the common dataflow operators like *Map*, *Reduce*, *Join*, etc.), which has been augmented with the *data-parallel*, *decentralized*, *asynchronous* ANN operator *Model*, with **train** and **predict** as two new OL functions. TensAIR is a **TensorFlow** framework developed on top of **AIR**, which means that it can scale out both the training and prediction tasks of an ANN model to multiple compute nodes, either with or without **GPUs** associated with them.

<!--- Run TensAIR on Singularity container (with or without GPUs):
------------
- Install Singularity (https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps)
- Install MPI; preferably OpenMPI 4.0.5

```sh
  $cd /path/to/TensAIR/
  $cd Singularity
  $mpirun -np <no.of dataflows> singularity exec --nv --pwd /home/TensAIR/Release tensair.sif ./TensAIR <use-case abbreviation> <mini batch size> <maxGradBuffer> <Throughput> <Epochs> <Syncronization Fator> <GPUs per node>
```
- To run TensAIR on Singularity usign SLURM please check the following script Singularity/runSingularityContainer.sh
- To create your own Singularity container check Singularity/tensair-singularity.def 
)
--->


Build & Run:
------------
### Install Dependencies:

- MPI; preferably OpenMPI/MPICH2 

- cmake

- pybind11 

- mpi4py 

- TensorFlow C API; version 2.3 or above ([tutorial](https://www.tensorflow.org/install))

- [saved_model_cli](https://www.tensorflow.org/guide/saved_model#install_the_savedmodel_cli) (automatically installed with TensorFlow)

**Note**: saved_model_cli shall be on PATH when running TensAIR

  On Mac(intel): 
  ```sh
    # Instal MPI
    $brew install openmpi
    # Instal CMake
    $brew install cmake

    # Install TensorFlow C API
    $wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-2.5.0.tar.gz
    $mkdir tensorflow_c_api
    $tar -C tensorflow_c_api -xzf libtensorflow-cpu-darwin-x86_64-2.5.0.tar.gz
    $sudo ldconfig

    #install pybind, mpi4py, and tensorflow
    $pip install pybind11 mpi4py tensorflow
  ```
  On Mac(arm m1/m2): 
  ```sh
    # Instal MPI
    $brew install openmpi
    # Instal CMake
    $brew install cmake

    # Install TensorFlow C API for Linux arm (workaround to do not have to build it from source) (https://gist.github.com/wangjia184/f9ffb2782d0703ef3dbceec9b2bbc4b4?permalink_comment_id=4269188#gistcomment-4269188)
    $brew install libtensorflow
    $cd $(brew --prefix libtensorflow)/lib #go to location in which libtensorflow was installed
    # Create links of the libraries with names on mac standard
    $ln -s libtensorflow.2.9.1.dylib libtensorflow.so.2.9.1
    $ln -s libtensorflow.2.dylib libtensorflow.so.2
    $ln -s libtensorflow.dylib libtensorflow.so
    
    #install pybind and mpi4py, and tensorflow
    $pip install pybind11 mpi4py tensorflow_macos
  ```

  On Linux: 
  ```sh
    # Install MPI
    $sudo apt-get install mpich
    # Install MPI
    $sudo apt-get install cmake
  
    # Instal TensorFlow C API
    $wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.5.0.tar.gz
    $mkdir tensorflow_c_api
    $tar -C tensorflow_c_api -xzf libtensorflow-cpu-linux-x86_64-2.5.0.tar.gz
    $sudo ldconfig

    #install pybind and mpi4py
    $pip install pybind11 mpi4py tensorflow
  ```
### Build and Run TensAIR

- Set compilation parameters
  - Update link_directories and include_directories accordinding with TensorFlow C API, pybind, and mpi4py. (lines 6 to 9 in CMakeLists.txt)
```sh
  Edit /path/to/TensAIR/CMakeLists.txt
```

- Build the project
```sh
  $cd /path/to/TensAIR/
  $mkdir Release
  $cd Release
  $cmake ../
  $make all
```

- Run a use-case
```sh
  $mpirun -np <no.of dataflows> ./TensAIR <use-case abbreviation>
  (E.g., mpirun -np 4 ./TensAIR W2V)
  (E.g., mpirun -np 8 ./TensAIR CIFAR)
  ```  

- Run using Python Interface
  - Create python script with dataflow. (python_test.py) 
  ```py
    from mpi4py import MPI
    import tensair_py
    comm = MPI.COMM_WORLD 

    mini_batch_size=32

    #create EventGenerator operator
    event_generator = tensair_py.CIFAR_EventGenerator(mpi_comm=comm, tag=1, mini_batch_size=mini_batch_size, msg_sec=10000, epochs=5, train_data_file="../data/CIFAR/cifar-train.txt") 

    #calculate maximum Message managed by TensAIR
    inputMaxSize = 4+4+ (8*2) + (4*mini_batch_size) + (4*mini_batch_size*32*32*3) #size of message between event generator and TensAIR
    gradientsMaxSize = 8+4+4+4+(8*12) + (4*((64*10) + (10) + (3*3*3*32) + (32) + (3*3*32*64) + (64) + (3*3*64*64) + (64) + (1024*64) + (64))) #size of message between TensAIR ranks
    window_size_CIFAR = max(inputMaxSize, gradientsMaxSize) #max message size managed by TensAIR
    
    #create TensAIR operator
    model = tensair_py.TensAIR(mpi_comm=comm, tag=2, window_size=window_size_CIFAR, epochs=5, saved_model_dir="../data/CIFAR/python_interface/cifar.tf", eval_data_file="../data/CIFAR/cifar-evaluate.bytes") 

    operators = [event_generator, model] #list of operators in dataflow
    links = [[0,1],[1,1]] #how the operators are linked

    basic_dataflow = tensair_py.BasicDataflow(mpi_comm=comm, operators=operators, links=links) #create dataflow
    basic_dataflow.streamProcess() #init dataflow
    ```  
  - Run with Python
  ```sh
    $mpirun -np <no.of dataflows> python <python_script>

    (E.g., $cp ../python_interface/python_test.py ./
           $mpirun -np 4 python ./python_test.py)
    ```  

- Run using SLURM
```sh
  $srun -n <no.of dataflows> ./TensAIR <use-case abbreviation>
```

Run under *OVERCOMMIT* option
```sh
  $srun --overcommit -n <no.of dataflows> ./TensAIR <use-case abbreviation>
```

### Available usecases

 - Word2Vec (W2V)
 - CIFAR-10 (CIFAR)
