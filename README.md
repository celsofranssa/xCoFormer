# xCoFormer

### 1. Quick Start

```shell script
# clone the project 
git clone git@github.com:Ceceu/xCoFormerr.git

# change directory to project folder
cd xCoFormerr/

# Create a new virtual environment by choosing a Python interpreter 
# and making a ./venv directory to hold it:
virtualenv -p python3 ./venv

# activate the virtual environment using a shell-specific command:
source ./venv/bin/activate

# install dependecies
pip install -r requirements.txt

# (if you need) to exit virtualenv later:
deactivate
```

### 2. Datasets


### 3. Test Run
```
python xCoFormer.py model=rnn data=java_v01 data.batch_size=128 trainer.max_epochs=1
```