

if __name__ == '__main__':
    import numpy as np

    sig =  lambda x : 1/(1+np.exp(-x))

    print(f"{3*sig(4.94775181)*(1-sig(4.94775181))}")

