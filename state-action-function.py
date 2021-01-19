import numpy as np

def state_action_function(p, r, pi,q):
    gamma = 0.9
    while True:
        q1 = r.T + gamma * p * pi * q
        if (q1 == q).all():
            break
        q = q1
    print(np.around(q.T, 1))

if __name__=='__main__':
    #fb c1 c2 c3 sleep
    #fb1 quit fb2 study1 sleep study2 study3 pub
    #probability matrix(action->state)
    P_action_matrix=np.mat([[1,0,0,0,0],
                            [0,1,0,0,0],
                            [1,0,0,0,0],
                            [0,0,1,0,0],
                            [0,0,0,0,1],
                            [0,0,0,1,0],
                            [0,0,0,0,1],
                            [0,0.2,0.4,0.4,0]])
    print('probability matrix(action->state):\n',P_action_matrix)
    #reward matrix(action)
    R_action_matrix=np.mat([-1,0,-1,-2,0,-2,10,1])
    print('reward matrix(action):\n',R_action_matrix)
    #policy matrix(state->action)
    pi_as_matrix=np.mat([[0.5,0.5,0,0,0,0,0,0],
                         [0,0,0.5,0.5,0,0,0,0],
                         [0,0,0,0,0.5,0.5,0,0],
                         [0,0,0,0,0,0,0.5,0.5],
                         [0,0,0,0,0,0,0,0]])
    print('policy matrix(state->action):\n',pi_as_matrix)
    #initialize q matrix
    q_matrix = np.mat(np.zeros((8, 1)))
    print('initialized q matrix:\n',q_matrix)
    print('state-action-function(fb1 quit fb2 study1 sleep study2 study3 pub):')
    state_action_function(P_action_matrix,R_action_matrix,pi_as_matrix,q_matrix)