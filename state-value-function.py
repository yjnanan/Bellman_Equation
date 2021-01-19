import numpy as np

def state_value_function(p,r,v):
    gamma=0.9
    while True:
        value=r.T+gamma*p*v
        if(value==v).all():
            break
        v=value
    return v

if __name__ =='__main__':
    #c1 c2 c3 pass pub fb sleep
    #probability matrix
    P_matrix=np.mat([[0,0.5,0,0,0,0.5,0],
                     [0,0,0.8,0,0,0,0.2],
                     [0,0,0,0.6,0.4,0,0],
                     [0,0,0,0,0,0,1.0],
                     [0.2,0.4,0.4,0,0,0,0],
                     [0.1,0,0,0,0,0.9,0],
                     [0,0,0,0,0,0,1]])
    print('probability matrix:\n',P_matrix)
    #reward matrix
    R_matrix=np.mat([-2,-2,-2,10,1,-1,0])
    print('reward matrix:\n',R_matrix)
    #initialize value matrix
    v_function=np.mat(np.zeros((7,1)))
    print('initialized value matrix:\n',v_function)
    print('value of each state(class1 class2 class3 pass pub facebook sleep:')
    print(np.around((state_value_function(P_matrix,R_matrix,v_function)).T,1))