import sys
import numpy as np

def test_matmul_error():
    X = np.ones((10, 2))
    
    # Is it scalar?
    R = np.float64(5.0)
    try:
        X @ R
    except Exception as e:
        print(f"Error for scalar R: {repr(e)}")
        
    # Is it tuple?
    try:
        X @ (1, 2)
    except Exception as e:
        print(f"Error for tuple R: {repr(e)}")
        
    # If R is 1D array
    try:
        X @ np.array([1, 2])
    except Exception as e:
        print(f"Error for 1D R: {repr(e)}")

with open("debug_log.txt", "w") as f:
    orig_stdout = sys.stdout
    sys.stdout = f
    test_matmul_error()
    sys.stdout = orig_stdout
