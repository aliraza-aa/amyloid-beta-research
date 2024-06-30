import matplotlib_inline.backend_inline
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib as mpl
from simtk.openmm import app
from simtk import openmm, unit
import mdtraj as md
from main import BlockAnalysis
from localcider.sequenceParameters import SequenceParameters
from numpy import linalg
from scipy.optimize import curve_fit
import scipy.stats as scs
import pandas as pd
import itertools
import numpy as np
import os
import shutil
import subprocess


def genDCD(name, eqsteps=10):
    """ 
    Generates coordinate and trajectory
    in convenient formats
    """
    traj = md.load("{:s}/pretraj.dcd".format(name),
                   top="{:s}/top.pdb".format(name))
    traj = traj.image_molecules(inplace=False, anchor_molecules=[
                                set(traj.top.chain(0).atoms)], make_whole=True)
    traj.center_coordinates()
    traj.xyz += traj.unitcell_lengths[0, 0]/2
    traj[int(eqsteps):].save_dcd("{:s}/traj.dcd".format(name))


genDCD("ab40")
