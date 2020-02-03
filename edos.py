# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 19:44:44 2020

@author: joeda
"""
import numpy as np
from numba import jit

@jit
def an(V) :
    return (0.1-0.01*V)/(np.exp(1-0.1*V)-1)
@jit
def am(V) :
    return (2.5-0.1*V)/(np.exp(2.5-0.1*V)-1)
@jit
def ah(V) :
    return 0.07*np.exp(-V/20)
@jit
def bn(V) :
    return 0.125*np.exp(-V/80)
@jit
def bm(V) :
    return 4*np.exp(-V/18)
@jit
def bh(V) :
    return 1/(np.exp(3-0.1*V)+1)