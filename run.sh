# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer/

# BERT PYTHON
python main.py \
  tasks=[fit] \
  model=BERT_TaG \
  data=PYTHON \
  data.folds=[0]


