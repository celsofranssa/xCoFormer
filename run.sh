# Create a new virtual environment by choosing a Python interpreter
# and making a ./venv directory to hold it:
virtualenv -p python3 ./venv

# activate the virtual environment using a shell-specific command:
source ./venv/bin/activate

# install dependecies
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# setting python path
export PYTHONPATH=$PATHONPATH:$PWD

# fit, predict, eval
python main.py \
  tasks=[fit,predict,eval] \
  model=CLM \
  data=JAVA \
  data.batch_size=64 \
  data.folds=[0,1,2,3,4] \
  data.num_workers=8

