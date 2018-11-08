import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

import nengo
import nengolib
from nengolib.signal import s, z

import os

figdir = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '../figures')

def savefig(name):
    plt.savefig(os.path.join(figdir, name), dpi=600, bbox_inches='tight')
