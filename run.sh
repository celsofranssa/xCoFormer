# activate venv and set Python path
source ~/projects/venvs/xCoFormer/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer/

# CLM_TaG over JAVASCRIPT with LR
python main.py \
  tasks=[zs_predict] \
  model=ZS_CLM \
  data=JAVASCRIPT \
  data.folds=[0]


