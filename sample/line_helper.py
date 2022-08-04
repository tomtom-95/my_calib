import numpy as np

def get_line_coeffs(line):
    """
    ax + by + c = 0
    """
    a = line[1,1] - line[0,1]
    b = line[0,0] - line[1,0]
    c = line[1,0]*line[0,1] - line[0,0]*line[1,1]

    return a, b, c

def get_line_params(line):
    """
    y = mx + q
    """
    m = (line[0,1] - line[1,1])/(line[0,0] - line[1,0])
    q = line[0,1] - m * line[0,0]

    return m, q
