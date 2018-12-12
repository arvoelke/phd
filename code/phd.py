from collections import defaultdict

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame

import nengo
import nengolib
from nengolib.signal import s, z
from nengolib.networks import RollingWindow

import os

figdir = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '../figures')

datadir = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '../data')


def savefig(name):
    plt.savefig(os.path.join(figdir, name), dpi=600, bbox_inches='tight')


def datapath(name):
    return os.path.join(datadir, name)
