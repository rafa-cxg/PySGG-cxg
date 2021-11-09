import numpy as np
import  os

from numba.core.serialize import pickle

fuck1=(1,3,10,11,12)
fuck2=[1,3,10,11,12]
fuck3={0:[1,2,3],
       1:[4,5,6],}
fuck4=[1,3]
shit={}
for key, value in fuck3.items():
    for v in value:
        shit[v]=key

exp=np.array((1,3))
c=np.array(list(fuck3.values()))
d=[shit[f] for f in fuck4]
print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__)))
with open(os.path.join("checkpointrepeat_dict.pkl"), 'rb') as f:
    pre_compute_bbox = pickle.load(f)
a=0
