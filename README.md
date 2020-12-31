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
After downloading the datasets from [Kaggle Datasets](https://www.kaggle.com/aldebbaran/code-search-datasets ), it should be placed inside the resources / datasets / folder as shown below:

```
xCoFormer/
|-- resources
|   |-- datasets
|   |   |-- java_v01
|   |   |   |-- test.jsonl
|   |   |   |-- train.jsonl
|   |   |   `-- val.jsonl
|   |   `-- python_v01
|   |       |-- test.jsonl
|   |       |-- train.jsonl
|   |       `-- val.jsonl
```

### 3. Test Run
```
python xCoFormer.py model=rnn data=java_v01 data.batch_size=128 trainer.max_epochs=1
```