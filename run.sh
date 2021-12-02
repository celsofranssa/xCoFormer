# activate venv and set Python path
source ~/projects/venvs/xCoFormer/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer/

# LSTM PYTHON
python main.py \
  tasks=[fit,predict,eval] \
  model=LSTM \
  data=PYTHON \
  data.folds=[0] \
  data.batch_size=128 \
  data.num_workers=8 \
  trainer.max_epochs=16 \
  trainer.patience=7 \
  trainer.min_delta=0.03


