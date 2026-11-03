"""Draws figure fig_A_F1: daily mean NO2 in the ten cities in 2016, CMAQ and the reference U-Net, in the form and range of fig_F3."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fig_F3 as F3
if __name__ == '__main__':
    F3.build('P', 'fig_A_F1')
