source env/bin/activate
export CUDA_VISIBLE_DEVICES=$1
export REM_ROOT=`pwd`
export PYTHONPATH=$REM_ROOT:$PYTHONPATH
export PYTHON_EXEC=python
export HUGGINGFACE_HUB_CACHE=/fruitbasket/users/dtredsox/huggingface_cache