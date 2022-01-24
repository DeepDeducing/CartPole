import numpy as np
import gym
import copy

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

#-------------------------------------------------------------------




stat_list = list()
for trials in range(100):                                                  # <<<<<<<<<<<<

    final_reward           = 0




    from Brain_for_deducing import *                                       # <<<<<<<<<<<<
    network_size           = np.array([4*100 + 2 * 25, 100, 100, 100, 64]) # <<<<<<<<<<<<
    beta                   = 1                                             # <<<<<<<<<<<<
    epoch_of_deducing      = 2000                                          # <<<<<<<<<<<<
    drop_rate              = 0.75                                          # <<<<<<<<<<<<
    Machine                = Brain(network_size, beta, epoch_of_deducing, drop_rate)

    weight_lists = list()
    slope_lists  = list()

    n_sets = 20                                                            # <<<<<<<<<<<<
    for n in range(n_sets):
        weight_name        = "100x100x100_25_0.000001_0.1m_0.2_[" + str(0 + n + 1) +  "]_weight_list.npy"   # <<<<<<<<<<<<
        slope_name         = "100x100x100_25_0.000001_0.1m_0.2_[" + str(0 + n + 1) +  "]_slope_list.npy"    # <<<<<<<<<<<<
        weight_list        = np.load(weight_name  , allow_pickle=True)
        slope_list         = np.load(slope_name   , allow_pickle=True)
        weight_lists.append(weight_list)
        slope_lists.append(slope_list)
    n_sets = 20                                                            # <<<<<<<<<<<<
    for n in range(n_sets):
        weight_name        = "100x100x100_25_0.000001_0.1m_0.2_[" + str(100 + n + 1) +  "]_weight_list.npy"   # <<<<<<<<<<<<
        slope_name         = "100x100x100_25_0.000001_0.1m_0.2_[" + str(100 + n + 1) +  "]_slope_list.npy"    # <<<<<<<<<<<<
        weight_list        = np.load(weight_name  , allow_pickle=True)
        slope_list         = np.load(slope_name   , allow_pickle=True)
        weight_lists.append(weight_list)
        slope_lists.append(slope_list)



    env                    = gym.make('CartPole-v0')                          # <<<<<<<<<<<<
    env._max_episode_steps = 999                                              # <<<<<<<<<<<<
    state                  = env.reset()
    env.render()                                                              # <<<<<<<<<<<<




    state_0        = quantifying(100, -2.5  , 0.050  , state[0])              # <<<<<<<<<<<<
    state_1        = quantifying(100, -3.75 , 0.075  , state[1])              # <<<<<<<<<<<<
    state_2        = quantifying(100, -0.375, 0.0075 , state[2])              # <<<<<<<<<<<<
    state_3        = quantifying(100, -3.75 , 0.075  , state[3])              # <<<<<<<<<<<<




    for t in range(10000):                                                    # <<<<<<<<<<<<
        print(t)                                                              # <<<<<<<<<<<<




        state_value                              = np.atleast_2d(inverse_sigmoid( np.concatenate((state_0, state_1, state_2, state_3)) ) ) # <<<<<<<<<<<<
        action_value                             = np.atleast_2d((np.random.random((1, 2*25)) - 0.5) * 0.00 - 3.5 )      # <<<<<<<<<<<<
        state_and_acton_value                    = np.atleast_2d( np.concatenate(( state_value[0], action_value[0] )) )
        state_and_acton_value_resistor           = np.zeros_like(state_and_acton_value)
        state_and_acton_value_resistor[:, 400:]  = 1                                                                     # <<<<<<<<<<<<
        reward                                   = np.ones(100)                                                          # <<<<<<<<<<<<
        for i in range(epoch_of_deducing):
            random_index         = np.random.randint(np.array(weight_lists).shape[0])
            weight_list          = weight_lists[random_index]
            slope_list           = slope_lists[random_index]
            Machine.network_size = np.array([weight_list[0].shape[0], 100, 100, 100, 100])
            state_and_acton_value[:, :  weight_list[0].shape[0] ]  = Machine.deduce_batch(state_and_acton_value[:, :  weight_list[0].shape[0] ],
                                                                                          state_and_acton_value_resistor[:, :  weight_list[0].shape[0] ],
                                                                                          reward,
                                                                                          weight_list, slope_list)
        action_value = state_and_acton_value[:, 400:]                                                                     # <<<<<<<<<<<<




        if np.argmax(action_value[0 , 0:2]) == 1:
            decided_action = int(1)
        if np.argmax(action_value[0 , 0:2]) == 0:
            decided_action = int(0)

        action = decided_action
        state, reward, done, info = env.step(action)
        env.render()                                     # <<<<<<<<<<<<

        state_0        = quantifying(100, -2.5  , 0.050  , state[0]) # <<<<<<<<<<<<
        state_1        = quantifying(100, -3.75 , 0.075  , state[1]) # <<<<<<<<<<<<
        state_2        = quantifying(100, -0.375, 0.0075 , state[2]) # <<<<<<<<<<<<
        state_3        = quantifying(100, -3.75 , 0.075  , state[3]) # <<<<<<<<<<<<




        final_reward += reward




        if done:                                         # <<<<<<<<<<<<
            print("Episode finished after {} timesteps".format(t + 1))
            print("Final reward:", final_reward)
            stat_list.append(final_reward)
            break




    env.close()




print("Average:", sum(stat_list)/len(stat_list))
print("Std:", np.std(stat_list))

