# activate venv and set Python path
source ~/projects/venvs/xCoFormer/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer/

# RoCLM_CoTNG PYTHON
python main.py \
  tasks=[fit] \
  model=FNet_CoTNG \
  data=JAVA \
  data.folds=[0] \
  data.batch_size=64 \
  data.num_workers=64 \
  trainer.max_epochs=16 \
  trainer.patience=7 \
  trainer.min_delta=0.03 \
  trainer.precision=32 \
  trainer.max_steps=1000
