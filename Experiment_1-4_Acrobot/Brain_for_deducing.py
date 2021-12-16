import numpy as np
from scipy.special import expit


class Brain(object):
    def __init__(self, network_size, beta, epoch_of_deducing, drop_rate):

        self.network_size                 = network_size
        self.number_of_layers             = self.network_size.shape[0]

        self.beta                         = beta
        self.epoch_of_deducing            = epoch_of_deducing

        self.drop_rate                    = drop_rate


    def activator(self, x):
        return expit(x)


    def activator_output_to_derivative(self, output):
        return output * ( 1 - output)



    def generate_values_for_each_layer(self, input):

        layer_list                = list()

        layer                     = input

        layer_list.append(layer)

        binomial                  = np.atleast_2d(np.random.binomial(1, 1 - self.drop_rate, size=self.network_size[1]))

        layer                     = self.activator(np.dot(layer_list[-1]                          , self.weight_list[0]                                                          ) * self.slope_list[0] ) * binomial

        layer_list.append(layer)

        for i in range(self.number_of_layers - 3):

            binomial              = np.atleast_2d(np.random.binomial(1, 1 - self.drop_rate, size=self.network_size[i + 2]))

            layer                 = self.activator(np.dot(layer_list[-1]                          , self.weight_list[i + 1]                                                      ) * self.slope_list[i + 1] ) * binomial

            layer_list.append(layer)

        layer                 = self.activator(np.dot(layer_list[-1]                          , self.weight_list[-1]                                                      ) * self.slope_list[-1] )

        layer_list.append(layer)

        return   layer_list


    def train_for_input_value(self,
                       layer_list, corresponding_output):

        layer_final_error      = corresponding_output - layer_list[-1]

        layer_delta            = layer_final_error                                                                                              * self.activator_output_to_derivative(layer_list[-1])           * self.slope_list[-1]

        for i in range(self.number_of_layers - 2):

            layer_delta        = (layer_delta.dot( self.weight_list[- 1 - i].T                                                          ) )     * self.activator_output_to_derivative(layer_list[- 1 - 1 - i])  * self.slope_list[-1 -1 -i]

        layer_delta        = (layer_delta.dot( self.weight_list[0].T                                                                    ) )     * self.activator_output_to_derivative(layer_list[0])

        self.known_and_unkown_input_value  += layer_delta  * self.beta    * self.known_and_unkown_input_value_resistor


    def deduce_batch(self, known_and_unkown_input_value, known_and_unkown_input_value_resistor, corresponding_output, weight_list, slope_list):


        self.weight_list    = weight_list
        self.slope_list     = slope_list


        self.known_and_unkown_input_value          = known_and_unkown_input_value
        self.known_and_unkown_input_value_resistor = known_and_unkown_input_value_resistor


        layer_list = self.generate_values_for_each_layer(self.activator( self.known_and_unkown_input_value ))
        self.train_for_input_value(layer_list, corresponding_output)


        return self.known_and_unkown_input_value


