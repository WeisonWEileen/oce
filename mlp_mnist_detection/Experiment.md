### environment setip
my host machine
```
4080 super, ubuntu 20.04, x86_64
```
conda env 
```
conda create -n clearlab_oc python=3.10 -y
conda activate clearlab_oc
pip install -r requirements.txt
```
### reproduce Resmlp with patch masking
please contact me if you cannot reproduce the result. I will use `.py` to implement the model because `.ipynb` is extremely slow.
```
cd mlp_mnist_detection
python3 main.py
```
### reproduce mlp baseline
```
cd mlp_mnist_detection
python3 baseline.py
```