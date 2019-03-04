# Author : Bayu Aditya

def levensthein(s, t):
    '''
        S : Source (key)
        T : Target
    '''

    import numpy as np
    s, t = s.upper(), t.upper()
    x = np.zeros((len(t)+1,len(s)+1))
    for i in range(len(t)+1):
        for j in range(len(s)+1):
            x[0,j] = j
            x[i,0] = i
    for i in range(1, len(t)+1):
        for j in range(1, len(s)+1):
            cost = (0 if (s[j-1] == t[i-1]) else 1)
            a = x[i-1, j] + 1
            b = x[i, j-1] + 1
            c = x[i-1,j-1] + cost
            x[i,j] = np.min([a,b,c])
    #print(x)
    return x[len(t), len(s)]
