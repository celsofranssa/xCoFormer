# activate venv and set Python path
source ~/projects/venvs/xCoFormer/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer/

# CLM_CoTNG JAVASCRIPT
python main.py \
  tasks=[z_shot] \
  model=CLM_CoTNG \
  data=JAVASCRIPT

# CLM_CoTNG PYTHON
python main.py \
  tasks=[z_shot] \
  model=CLM_CoTNG \
  data=PYTHON

# CLM_CoTNG JAVA
python main.py \
  tasks=[z_shot] \
  model=CLM_CoTNG \
  data=JAVA
