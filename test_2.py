from utils import NetMCData
from pathlib import Path
from os import scandir
import numpy as np

a = np.array([[1,2,3],[4,5,6]])
b = np.array([7,8,9])
c = np.vstack((a,b))
print(c)