source ~/.virtualenvs/diversity-recommendations/bin/activate

export PYTHON_PROGRAM=$1
export PROGRAM_CONFIG=$2

python3 $PYTHON_PROGRAM $PROGRAM_CONFIG

deactivate