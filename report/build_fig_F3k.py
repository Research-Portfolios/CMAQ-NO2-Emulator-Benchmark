"""Draws figure fig_F3k: each U-Net's city mean of its daily differences from CMAQ in the ten cities of fig_F3 in 2016, on a pale bar between two percentiles of those differences, with the city series and the drawing parts of build_fig_F3.py and build_port_fig_F3.py."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fig_F3 as F3
if __name__ == '__main__':
    F3.build_k('fig_F3k')
