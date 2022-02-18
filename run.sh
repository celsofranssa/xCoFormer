# activate venv and set Python path
source ~/projects/venvs/xCoFormer/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer/

# RoCLM_CoTNG PYTHON
python main.py \
  tasks=[fit,predict,eval] \
  model=CLM_CoTNG \
  data=JAVA \
  data.batch_size=64 \
  data.num_workers=12
