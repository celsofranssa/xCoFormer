# activate venv and set Python path
source /home/celso/projects/venvs/xCoFormer/bin/activate
export PYTHONPATH=$PATHONPATH:/home/celso/projects/xCoFormer/

# BiLSTM
python main.py \
  tasks=[fit,predict,eval] \
  model=BiLSTM \
  data=PYTHON \
  data.batch_size=64 \
  data.num_workers=64 \
  trainer.max_epochs=64 \
  trainer.patience=31 \
  trainer.min_delta=0.01

# SelfAtt
python main.py \
  tasks=[fit,predict,eval] \
  model=SelfAtt \
  data=PYTHON \
  data.batch_size=64 \
  data.num_workers=64 \
  trainer.max_epochs=64 \
  trainer.patience=31 \
  trainer.min_delta=0.01


