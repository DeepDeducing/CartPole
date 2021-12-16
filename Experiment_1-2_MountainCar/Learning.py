import numpy as np
import gym
import math

""" Please note that, the initial state in each iteration is set to:
        self.state = np.array([self.np_random.uniform(low= - 0.6, high= 0.6), self.np_random.uniform(low= - 0.1, high= 0.1)]) """

#--------------------------------------------------------------------

def inverse_sigmoid(input):
    return np.log((input+ 0.0000000001) /(1-input + 0.0000000001))

def vectorizing(array_size, init, interv, input):
    array = np.zeros(array_size)
    array[int(array_size//2 - 1 + (input - init) // interv)] = 1
    return array

def quantifying(array_size, init, interval, input):
    array = np.zeros(array_size)
    if int( (input - init) // interval + 1) >= 0:
        array[ : int( (input - init) // interval + 1)] = 1
    return array

#--------------------------------------------------------------------

start_set     = 1   # <<<<<<<<<<<<
end_set       = 5   # <<<<<<<<<<<<

n_sets        = end_set - start_set + 1

for n in range(n_sets):




    from Brain_for_learning import *
    network_size              = np.array([100 * 2  + 3 * 70, 100, 100, 100, 100])  # <<<<<<<<<<<<
    slope                     = 25                                                 # <<<<<<<<<<<<
    alpha                     = 0.000001                                           # <<<<<<<<<<<<
    epoch_of_learning         = 4000000                                            # <<<<<<<<<<<<
    drop_rate                 = 0.2                                                # <<<<<<<<<<<<
    momentum_rate             = 0.9                                                # <<<<<<<<<<<<

    Machine                   = Brain(network_size, slope, alpha, epoch_of_learning, drop_rate, momentum_rate)

    retrain = False                                                                # <<<<<<<<<<<<
    if retrain == True:
        Machine.weight_list            = np.load("100x100x100_25_0.000001_4m_0.2_[" + str(start_set + n) +  "]_weight_list.npy"          , allow_pickle=True)
        Machine.slope_list             = np.load("100x100x100_25_0.000001_4m_0.2_[" + str(start_set + n) +  "]_slope_list.npy"           , allow_pickle=True)
        Machine.weight_list_momentum   = np.load("100x100x100_25_0.000001_4m_0.2_[" + str(start_set + n) +  "]_weight_list_momentum.npy" , allow_pickle=True)
        Machine.slope_list_momentum    = np.load("100x100x100_25_0.000001_4m_0.2_[" + str(start_set + n) +  "]_slope_list_momentum.npy"  , allow_pickle=True)




    for i_episode in range(epoch_of_learning):

        print(i_episode)




        final_reward         = 0




        env                  = gym.make('MountainCar-v0')                       # <<<<<<<<<<<<
        state                = env.reset()
        #env.render()                                                           # <<<<<<<<<<<<




        random_initial_moves = np.random.randint(10)                  # <<<<<<<<<<<<
        for t in range(random_initial_moves):                         # <<<<<<<<<<<<
            action                    = env.action_space.sample()
            state, reward, done, info = env.step(action)
            # env.render()                                            # <<<<<<<<<<<<




        state_0        = quantifying(100, -0.6  , 0.012  , state[0])  # <<<<<<<<<<<<
        state_1        = quantifying(100, -0.1  , 0.002  , state[1])  # <<<<<<<<<<<<




        action_list = np.zeros(3 * 70)  # <<<<<<<<<<<<
        for t in range(70):             # <<<<<<<<<<<<
            action                          = env.action_space.sample()
            state, reward, done, info       = env.step(action)
            #env.render()               # <<<<<<<<<<<<
            action_list[t * 3 + action]     = 1
            final_reward                   += reward
            #if done:                   # <<<<<<<<<<<<
            #    break




        reward               = quantifying(100, -0.6, 0.012, state[0])               # <<<<<<<<<<<<




        Machine.learn_batch(       np.atleast_2d(           np.concatenate((state_0, # <<<<<<<<<<<<
                                                            state_1,
                                                            action_list
                                                            )) ),
                                            np.array([reward])  )




    env.close()




    np.save("100x100x100_25_0.000001_4m_0.2_[" + str(start_set + n) +  "]_weight_list"             , Machine.weight_list                 ) # <<<<<<<<<<<<
    np.save("100x100x100_25_0.000001_4m_0.2_[" + str(start_set + n) +  "]_slope_list"              , Machine.slope_list                  ) # <<<<<<<<<<<<
    np.save("100x100x100_25_0.000001_4m_0.2_[" + str(start_set + n) +  "]_weight_list_momentum"    , Machine.weight_list_momentum        ) # <<<<<<<<<<<<
    np.save("100x100x100_25_0.000001_4m_0.2_[" + str(start_set + n) +  "]_slope_list_momentum"     , Machine.slope_list_momentum         ) # <<<<<<<<<<<<





