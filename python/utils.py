import random

def create_random_int_arrays(amount_in_a, amount_in_b) :
    tab = list(range(0, amount_in_a + amount_in_b))

    tab_a = []
    tab_b = []

    for i in range(0, amount_in_a):
        index = random.randint(0, len(tab)-1)
        tab_a.append(tab[index])
        del tab[index]

        
    for i in range(0, amount_in_b):
        index = random.randint(0, len(tab)-1)
        tab_b.append(tab[index])
        del tab[index]

    return (tab_a, tab_b)
