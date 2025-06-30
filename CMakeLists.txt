cmake_minimum_required(VERSION 3.15)
project(TensAIR)

set(CMAKE_CXX_STANDARD 17)

# Access environment variables
set(PYBIND_INCLUDE_DIR $ENV{PYBIND_INCLUDE_DIR})
#set(PYBIND_INCLUDE_DIR ".../python3.9/site-packages/pybind11/include") #pip3 show pybind11
set(MPI4PY_INCLUDE_DIR $ENV{MPI4PY_INCLUDE_DIR})
#set(MPI4PY_INCLUDE_DIR ".../python3.9/site-packages/mpi4py/include") #pip3 show mpi4py
set(LIBTENSORFLOW_LIB_DIR $ENV{LIBTENSORFLOW_LIB_DIR})
#set(LIBTENSORFLOW_LIB_DIR ".../libtensorflow/lib") #tf_c location/lib
set(LIBTENSORFLOW_INCLUDE_DIR $ENV{LIBTENSORFLOW_INCLUDE_DIR})
#set(LIBTENSORFLOW_INCLUDE_DIR ".../libtensorflow/include") #tf_c location/include

#find required packeges
find_package(MPI 3 REQUIRED) #find MPI
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED) #find Python, set manually with the python path in "python3 -m pybind11 --includes"

#Include external directories
include_directories(${LIBTENSORFLOW_INCLUDE_DIR}) #tensorflow_c_api
include_directories(${PYBIND_INCLUDE_DIR})        #pybind
include_directories(${Python_INCLUDE_DIRS})       #python
include_directories(${MPI4PY_INCLUDE_DIR})        #mpi4py
include_directories(${MPI_CXX_INCLUDE_DIRS})      #mpi

#Link Tensorflow
link_directories(${LIBTENSORFLOW_LIB_DIR}) # tensorflow_c_api lib directory 

# Find the saved_model_cli executable
find_program(SAVED_MODEL_CLI_EXECUTABLE NAMES saved_model_cli REQUIRED)

# Check if the saved_model_cli was found
if(NOT SAVED_MODEL_CLI_EXECUTABLE)
    message(FATAL_ERROR "saved_model_cli not found. Please ensure it's installed and in your PATH.")
else()
    message(STATUS "saved_model_cli found at: ${SAVED_MODEL_CLI_EXECUTABLE}")
endif()

#Macro that lists all internal directories which have a .hpp file
MACRO(HEADER_DIRECTORIES return_list)
    FILE(GLOB_RECURSE new_list *.hpp)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()
HEADER_DIRECTORIES(header_dir_list)

#Include internal directories
include_directories(${header_dir_list})

# List of all .cpp and .hpp used by TensAIR (Manually set)
set(LIST_FILES
        src/batching/Map.cpp
        src/batching/Map.hpp
        src/batching/Reduce.cpp
        src/batching/Reduce.hpp
        src/collector/BinCollector.cpp
        src/collector/BinCollector.hpp
        src/collector/RowCollector.cpp
        src/collector/RowCollector.hpp
        src/communication/Message.cpp
        src/communication/Message.hpp
        src/communication/Window.cpp
        src/communication/Window.hpp
        src/configuration/HashConfig.hpp
        src/connector/BinConnector.cpp
        src/connector/BinConnector.hpp
        src/connector/PartRowConnector.cpp
        src/connector/PartRowConnector.hpp
        src/connector/RowConnector.cpp
        src/connector/RowConnector.hpp
        src/dataflow/Dataflow.cpp
        src/dataflow/Dataflow.hpp
        src/dataflow/BasicDataflow.hpp
        src/dataflow/Vertex.cpp
        src/dataflow/Vertex.hpp
        src/dataflow/BasicVertex.hpp
        src/dataflow/CountAggregator.hpp
        src/dataflow/FlowWrappingAggregator.hpp
        src/dataflow/SortAggregator.hpp
        src/function/Function.cpp
        src/function/Function.hpp
        src/function/SquareFunction.cpp
        src/function/SquareFunction.hpp
        src/function/SumFunction.cpp
        src/function/SumFunction.hpp
        src/input/FileInput.cpp
        src/input/FileInput.hpp
        src/input/Input.cpp
        src/input/Input.hpp
        src/main/main.cpp
        src/output/FileOutput.cpp
        src/output/FileOutput.hpp
        src/output/Output.cpp
        src/output/Output.hpp
        src/partitioning/Hash.cpp
        src/partitioning/Hash.hpp
        src/partitioning/PairPartition.hpp
        src/partitioning/Partition.hpp
        src/relational/Aggr.cpp
        src/relational/Aggr.hpp
        src/relational/Attribute.cpp
        src/relational/Attribute.hpp
        src/relational/Cond.cpp
        src/relational/Cond.hpp
        src/relational/Relation.cpp
        src/relational/Relation.hpp
        src/relational/Schema.cpp
        src/relational/Schema.hpp
        src/relational/SortAggregation.cpp
        src/relational/SortAggregation.hpp
        src/relational/SortMergeJoin.cpp
        src/relational/SortMergeJoin.hpp
        src/serialization/Serialization.cpp
        src/serialization/Serialization.hpp
        src/TensAIR/TensAIR.hpp
        src/TensAIR/TensAIR.cpp
        src/TensAIR/DriftDetector.hpp
        src/TensAIR/DriftDetector.cpp
        src/TensAIR/CircularList.cpp
        src/usecases/MapReduce.cpp
        src/usecases/MapReduce.hpp
        src/usecases/StreamingTest.cpp
        src/usecases/StreamingTest.hpp
        src/usecases/WordEmbedding.hpp
        src/usecases/WordEmbedding.cpp
        src/usecases/Cifar.hpp
        src/usecases/Cifar.cpp
        src/usecases/Demo.hpp
        src/usecases/Demo.cpp
        src/usecases/VGG16.cpp
        src/usecases/VGG16.hpp
        src/usecases/RESNET50.cpp
        src/usecases/RESNET50.hpp
        src/word_embedding/EventGenerator.hpp
        src/word_embedding/EventGenerator.cpp
        src/cifar/EventGenerator.hpp
        src/cifar/EventGenerator.cpp
        src/demo/EventGenerator.hpp
        src/demo/EventGenerator.cpp
        src/vgg16_Convergence/VGG16_Convergence.cpp
        src/vgg16_Convergence/VGG16_Convergence.hpp
        src/resnet50_Convergence/RESNET50_Convergence.cpp
        src/resnet50_Convergence/RESNET50_Convergence.hpp
        src/airplane/EventGenerator.hpp
        src/airplane/EventGenerator.cpp
)

#Python Executable
add_executable(tensair_py.so ${LIST_FILES} src/TensAIR/Python_Interface.cpp src/TensAIR/EventGenerator.cpp src/usecases/Demo.cpp)
target_compile_definitions(tensair_py.so PRIVATE SAVED_MODEL_CLI_PATH="${SAVED_MODEL_CLI_EXECUTABLE}")
target_link_libraries(tensair_py.so ${MPI_LIBRARIES}) 
# Set the output directory for tensair_py.so
set_target_properties(tensair_py.so PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)
if(APPLE)
    target_compile_options(tensair_py.so PRIVATE -shared -fPIC -pthread -undefined dynamic_lookup)
    target_link_options(tensair_py.so PRIVATE -shared -fPIC -pthread -undefined dynamic_lookup)
elseif(UNIX AND NOT APPLE)
    target_compile_options(tensair_py.so PRIVATE -shared -fPIC -pthread)
    target_link_options(tensair_py.so PRIVATE -shared -fPIC -pthread)
else()
    message("ERROR! TensAIR was built for Unix based systems.")
endif()
target_link_libraries(tensair_py.so "-ltensorflow") 
target_link_libraries(tensair_py.so tensorflow)

#C++ executable
add_executable(TensAIR ${LIST_FILES}) 
target_compile_definitions(TensAIR PRIVATE SAVED_MODEL_CLI_PATH="${SAVED_MODEL_CLI_EXECUTABLE}")
target_link_libraries(TensAIR ${MPI_LIBRARIES})
target_link_libraries(TensAIR ${Python_LIBRARIES})
# Set the output directory for tensair_py.so
set_target_properties(TensAIR PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)
target_compile_options(TensAIR PRIVATE -pthread)
target_link_options(TensAIR PRIVATE -pthread)
target_link_libraries(TensAIR "-ltensorflow")
target_link_libraries(TensAIR tensorflow)

# Compute the absolute path to the desired directory
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(LIB_DIR "${PARENT_DIR}/lib")

# Prepend the computed directory to PYTHONPATH for processes launched by CMake
set(ENV{PYTHONPATH} "${LIB_DIR}:$ENV{PYTHONPATH}")
