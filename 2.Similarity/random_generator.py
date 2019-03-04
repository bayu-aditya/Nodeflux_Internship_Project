import numpy as np

def random_word_generator():
    list_word = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    loc = np.random.randint(low = 0, high=len(list_word))
    result = list_word[loc]
    return result

def random_number_generator():
    list_number = '0123456789'
    loc = np.random.randint(low = 0, high=len(list_number))
    result = list_number[loc]
    return result