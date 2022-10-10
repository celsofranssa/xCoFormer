# activate venv and set Python path
source ~/projects/venvs/xCoFormer/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer/

# CLM_TaG over JAVASCRIPT with LR
python main.py \
  tasks=[fit] \
  model=CLM \
  data=JAVASCRIPT \
  data.folds=[0]


