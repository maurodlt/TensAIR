#!/bin/bash

######################################################## mpi4py
# Get the location of mpi4py from pip3 show
MPI4PY_LOCATION=$(pip3 show mpi4py | grep "Location" | cut -d' ' -f2)
# Append "/mpi4py/include" to the location
DEFAULT_MPI4PY_INCLUDE_PATH="${MPI4PY_LOCATION}/mpi4py/include"
# Ask the user if they want to use the default path
echo "Detected mpi4py include path: ${DEFAULT_MPI4PY_INCLUDE_PATH}"
read -p "Do you want to use this path? [Y/n]: " USE_DEFAULT

if [[ $USE_DEFAULT =~ ^[Yy]$ ]] || [[ -z $USE_DEFAULT ]]; then
    # User chose to use the default or pressed enter
    MPI4PY_INCLUDE_PATH="${DEFAULT_MPI4PY_INCLUDE_PATH}"
else
    # User chose not to use the default, prompt for a new path
    read -p "Enter the mpi4py include path: " USER_SPECIFIED_PATH
    MPI4PY_INCLUDE_PATH="${USER_SPECIFIED_PATH}"
fi
# Export it as an environment variable
export MPI4PY_INCLUDE_DIR="${MPI4PY_INCLUDE_PATH}"


##################################################### pybind11
# Get the location of mpi4py from pip3 show
PYBIND_LOCATION=$(pip3 show pybind11 | grep "Location" | cut -d' ' -f2)
# Append "/mpi4py/include" to the location
DEFAULT_PYBIND_INCLUDE_PATH="${PYBIND_LOCATION}/pybind11/include"
# Ask the user if they want to use the default path
echo "Detected pybind11 include path: ${DEFAULT_PYBIND_INCLUDE_PATH}"
read -p "Do you want to use this path? [Y/n]: " USE_DEFAULT

if [[ $USE_DEFAULT =~ ^[Yy]$ ]] || [[ -z $USE_DEFAULT ]]; then
    # User chose to use the default or pressed enter
    PYBIND_INCLUDE_PATH="${DEFAULT_PYBIND_INCLUDE_PATH}"
else
    # User chose not to use the default, prompt for a new path
    read -p "Enter the mpi4py include path: " USER_SPECIFIED_PATH
    PYBIND_INCLUDE_PATH="${USER_SPECIFIED_PATH}"
fi
# Export it as an environment variable
export PYBIND_INCLUDE_DIR="${PYBIND_INCLUDE_PATH}"


################################################## tf_c location
IFS=':' read -ra LIBRARY_PATH_ENTRIES <<< "$LIBRARY_PATH"
COMMON_PATHS=(
    "/usr/local/lib"
    "/usr/lib"
    "$HOME/.local/lib"
    "/opt/tensorflow/lib"
    "${LIBRARY_PATH_ENTRIES[@]}" # Include paths from LIBRARY_PATH
)

# Attempt to find libtensorflow.so or libtensorflow.dylib in the common paths
FOUND_PATH=""
for path in "${COMMON_PATHS[@]}"; do
    #echo "Checking in: $path"
    if [ -f "$path/libtensorflow.so" ] || [ -f "$path/libtensorflow.dylib" ]; then
        FOUND_PATH="$path"
        #echo "Found libtensorflow in: $FOUND_PATH"
        break
    fi
done


if [ -z "$FOUND_PATH" ]; then
    # User cmanually sets a path as a default one was not located
    read -p "libtensorflow could not be located. You probably forgot to include it into the LIBRARY_PATH. Enter the path to libtensorflow manually (eg. /opt/libtensorflow): " USER_SPECIFIED_PATH
    LIBTENSORFLOW_PATH="${USER_SPECIFIED_PATH}"

else
    echo "Detected libtensorflow path: ${FOUND_PATH}"
    read -p "Do you want to use this path? [Y/n]: " USE_DEFAULT
    if [[ $USE_DEFAULT =~ ^[Yy]$ ]] || [[ -z $USE_DEFAULT ]]; then
        # User chose to use the default or pressed enter
        LIBTENSORFLOW_PATH="${FOUND_PATH}"
    else
        # User chose not to use the default, prompt for a new path
        read -p "Enter the directory of libtensorflow: " USER_SPECIFIED_PATH
        LIBTENSORFLOW_PATH="${USER_SPECIFIED_PATH}"
    fi
fi

# Export it as an environment variable
export LIBTENSORFLOW_LIB_DIR="${LIBTENSORFLOW_PATH}"
export LIBTENSORFLOW_INCLUDE_DIR="${LIBTENSORFLOW_PATH%????}/include"

# Create /lib folder if it does not exist
lib_folder="$(dirname "${BASH_SOURCE[0]}")/lib"
# Check if the folder doesn't exist
if [ ! -d "$lib_folder" ]; then
    # Create the folder
    mkdir -p "$lib_folder"
fi


#Export TensAIR location
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/lib" && pwd)"
INCLUDE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/include" && pwd)"
TensAIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" 
export PYTHONPATH="${LIB_DIR}:${PYTHONPATH}"
export PYTHONPATH="${INCLUDE_DIR}:${PYTHONPATH}"
export TENSAIR_PATH="${TensAIR_DIR}"

