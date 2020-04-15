from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        #seed the rng so it stays the same through runs
        random.seed(1)

        #create a 3x1 matrix and assign weights randomly
        #make sure the weights are between -1 and 1
        #this'll be the single neuron
        self.synaptic_weights = 2*random.random((4,)) - 1

    def __sigmoid(self, x):
        '''the sigmoid function
           it receives the weighted sum of the inputs and normalises them
           between 0 and 1
        '''
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_set_inputs, training_set_outputs, iterations):
        
        for iteration in range(iterations):
            #pass the training set through the neural net
            output = self.predict(training_set_inputs)

            #calculate the error
            error = training_set_outputs - output

            #multiply the error if the initial input by the gradient of the sigmoid curve
            #this is called "Gradient Descent"
            adjustment = dot(training_set_outputs.T, error * self.__sigmoid_derivative(output))

            #adjust the weights
            self.synaptic_weights += adjustment


    def predict(self, inputs):
         '''pass the inputs through our neuron(s)
            then pass the result to our sigmoid function
         '''
         return self.__sigmoid(dot(inputs, self.synaptic_weights))