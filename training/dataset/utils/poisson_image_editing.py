"""Poisson image editing.

"""

import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve

from os import path

def laplacian_matrix(n, m):
    """Generate the Poisson matrix. 

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A


def poisson_edit(source, target, mask, offset):
    pass

def main():    
    scr_dir = 'figs/example1'
    out_dir = scr_dir
    source = cv2.imread(path.join(scr_dir, "source1.jpg")) 
    target = cv2.imread(path.join(scr_dir, "target1.jpg"))    
    mask = cv2.imread(path.join(scr_dir, "mask1.png"), 
                      cv2.IMREAD_GRAYSCALE) 
    offset = (0,66)
    result = poisson_edit(source, target, mask, offset)

    cv2.imwrite(path.join(out_dir, "possion1.png"), result)
    

if __name__ == '__main__':
    main()
