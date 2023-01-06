y = [1,23,4,5,-34,-3,4,-7,-99,3]
print (y)

def remove_negs(y):
    r = y[:]
    for ampl in y:
        if ampl<0:
            r.remove(ampl)
    return r

y = remove_negs(y)
print(y)