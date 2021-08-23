import numpy as np
import gym
import copy

"Some basic functions to transfer real values to quantified vectors or one-hotted vectors"
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


cal_list = list()
for trials in range(100):


    "Importing neural network module and setting up parameters"
    from Brain_for_deducing import *

    network_size                = np.array([100 * 4 + 50, 100, 100, 100, 100])

    beta                        = 0.1

    epoch_of_deducing           = 5000

    drop_rate                   = 0.75

    Machine                     = Brain(network_size, beta, epoch_of_deducing, drop_rate)


    "Importing sets of trained weight matrices from the learning phase"
    weight_lists = list()
    slope_lists  = list()
    "M sets, controlling angle"
    n_sets = 5
    for j in range(n_sets):
        weight_name        = "100x100x100_25_0.000001_1m_0.2_[" + str(0 + j + 1) +  "]_weight_list.npy"
        slope_name         = "100x100x100_25_0.000001_1m_0.2_[" + str(0 + j + 1) +  "]_slope_list.npy"
        weight_list        = np.load(weight_name  , allow_pickle=True)
        slope_list         = np.load(slope_name   , allow_pickle=True)
        weight_lists.append(weight_list)
        slope_lists.append(slope_list)
    "M-K sets, controlling position"
    n_sets = 5
    for j in range(n_sets):
        weight_name        = "100x100x100_25_0.000001_1m_0.2_[" + str(100 + j + 1) +  "]_weight_list.npy"
        slope_name         = "100x100x100_25_0.000001_1m_0.2_[" + str(100 + j + 1) +  "]_slope_list.npy"
        weight_list        = np.load(weight_name  , allow_pickle=True)
        slope_list         = np.load(slope_name   , allow_pickle=True)
        weight_lists.append(weight_list)
        slope_lists.append(slope_list)


    "Importing environment"
    env                    = gym.make('CartPole-v0')
    env._max_episode_steps = 999
    state                  = env.reset()


    "Quantifying states"
    state_0        = quantifying(100, -2.5  , 0.050  , state[0])
    state_1        = quantifying(100, -3.75 , 0.075  , state[1])
    state_2        = quantifying(100, -0.375, 0.0075 , state[2])
    state_3        = quantifying(100, -3.75 , 0.075  , state[3])


    final_reward           = 0


    for t in range(10000):


        #env.render()


        "Setting up given states, initialized actions and optinal reward"
        state_value                              = np.atleast_2d(inverse_sigmoid(  np.concatenate((state_0, state_1, state_2, state_3))  ) )
        action_value                             = np.atleast_2d((np.random.random((1, 50)) - 0.5) * 0.0000 - 3.5 )
        state_and_acton_value                    = np.atleast_2d( np.concatenate(( state_value[0], action_value[0] )) )
        state_and_acton_value_resistor           = np.zeros_like(state_and_acton_value)
        state_and_acton_value_resistor[:, 400:]  = 1
        reward                                   = np.atleast_2d( np.ones(100) )


        "Start dedcuing by MWM-SGD and error backpropagation"
        for i in range(epoch_of_deducing):

            random_index = np.random.randint(np.array(weight_lists).shape[0])
            weight_list  = weight_lists[random_index]
            slope_list   = slope_lists[random_index]

            state_and_acton_value  = Machine.deduce_batch(state_and_acton_value,
                                                          state_and_acton_value_resistor,
                                                          reward,
                                                          weight_list, slope_list)



        action_value = state_and_acton_value[:, 400:]



        "Deciding real/final action based on optimzed initial actions"
        #print(action_value[0, 0:2])
        if np.argmax(action_value[0, 0:2]) == 1:
            decided_action = int(1)
        #    print("right")
        else:
            decided_action = int(0)
        #    print("left")


        "Return new state"
        action                          = decided_action
        state, reward, done, info = env.step(action)

        state_0        = quantifying(100, -2.5  , 0.050  , state[0])
        state_1        = quantifying(100, -3.75 , 0.075  , state[1])
        state_2        = quantifying(100, -0.375, 0.0075 , state[2])
        state_3        = quantifying(100, -3.75 , 0.075  , state[3])


        final_reward += reward
        if done:
           print("Episode finished after {} timesteps".format(t+1))
           print("Final reward:", final_reward)
           cal_list.append(final_reward)
           break


    env.close()


print("Average:", sum(cal_list)/len(cal_list))
print("Std:", np.std(cal_list))

