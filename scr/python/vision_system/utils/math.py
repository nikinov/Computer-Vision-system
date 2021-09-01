import numpy as np

# function to return the nearest value in a list
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# function to return key for any value
def get_key(my_dict, val):
    for key, value in my_dict.items():
         if val == value:
             return key