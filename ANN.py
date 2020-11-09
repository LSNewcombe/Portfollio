import numpy as np
import pandas as pd
import decimal as D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle as pic

class ValidationFunctions():

    @staticmethod
    def datacheck(x):
        # define the sigmoid activation function
        minval = 1000000
        maxval = -1000000
        minvalexp = 1000000
        maxvalexp = -10000000
        z = 0
        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j] >= 0:
                    z = np.exp(-x[i][j])
                    if -x[i][j] < minval:
                        minval = -x[i][j]

                else:
                    z = np.exp(x[i][j])
                    if x[i][j] > maxval:
                        maxval = x[i][j]

                if z > maxvalexp:
                    maxvalexp = z

                if z < minvalexp:
                    minvalexp = z

        return minvalexp, maxvalexp, minval, maxval


class ActivationFunctions():
    # ---------ORIGINAL PARAMETRISED ACTIVATION FUNCTIONS-------------------------------------------------
    @staticmethod
    def reLuDerivative(x):

        for i in range(len(x)):
            for j in range(len(x[i])):

                if x[i][j] < 0:
                    x[i][j] = 0

                elif x[i][j] >= 0:
                    x[i][j] = 1
        return x

    @staticmethod
    def sigmoid(x):
        # define the sigmoid activation function
        toExcel = pd.DataFrame(x)
        toExcel.to_excel(r'DotProduct.xlsx', sheet_name='Dot Product')

        for i in range(len(x)):
            for j in range(len(x[i])):

                if x[i][j] >= 0:
                    z = np.exp(-x[i][j])
                    w = 1 + z
                    x[i][j] = 1 / w
                elif x[i][j] < 0:
                    z = np.exp(x[i][j])
                    x[i][j] = z / (1 + z)

        toExcel = pd.DataFrame(x)
        toExcel.to_excel(r'Sigmoid.xlsx', sheet_name='Dot Product')

        return x

    @staticmethod
    def sigmoidDerivative(x):
        # calculate the derivative of sigmoid
        return x * (1 - x)

    @staticmethod
    def gSign(x, b):
        for i in range(len(x)):
            for j in range(len(x[i])):

                bi = D.Decimal(b[j])
                if x[i][j] < 0:
                    expo = D.Decimal(x[i][j]).exp()
                    x[i][j] = ((1 - bi) * expo - bi) / (1 + expo)
                else:
                    expo = D.Decimal(-x[i][j]).exp()
                    x[i][j] = ((1 - bi) - bi * expo) / (1 + expo)

        return x

    @staticmethod
    def sigmoidMM(x):
        toExcel = pd.DataFrame(x)
        toExcel.to_excel(r'DotProduct.xlsx', sheet_name='Dot Product')

        for i in range(len(x)):
            for j in range(len(x[i])):

                if x[i][j] < -33.3:
                    x[i][j] = x[i][j]

                elif x[i][j] <= -18:
                    x[i][j] = (x[i][j] - np.exp(-x[i][j]))

                elif x[i][j] <= 37:
                    x[i][j] = -np.log1p(np.exp(-x[i][j]))

                else:
                    x[i][j] = -np.exp(-x[i][j])

        toExcel = pd.DataFrame(x)
        toExcel.to_excel(r'Sigmoid.xlsx', sheet_name='Sigmoid')

        return x

    @staticmethod
    def sigmoidNaive(x):
        toExcel = pd.DataFrame(x)
        toExcel.to_excel(r'DotProduct.xlsx', sheet_name='Dot Product')

        for i in range(len(x)):
            for j in range(len(x[i])):

                if x[i][j] >= 0:

                    x[i][j] = np.log(1 / (1 + np.exp(-x[i][j])))

                elif x[i][j] < 0:
                    z = np.exp(x[i][j])
                    x[i][j] = np.log(z / (1 + z))

        toExcel = pd.DataFrame(x)
        toExcel.to_excel(r'Sigmoid.xlsx', sheet_name='Sigmoid')

        return x

    @staticmethod
    def tanh(x):

        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def tanhDerivative(x):

        return 1 - np.power(ActivationFunctions.tanh(x), 2)

# Represents the Storage of the Layer


class NeuronLayer():
    def __init__(self, InputTotal, neuronsTotal, layername):
       
        self.synapticWeights = np.random.randn(InputTotal, neuronsTotal) * np.sqrt(1 / (neuronsTotal))

        self.bias = np.ones(neuronsTotal)
        self.layeridname = layername
        toExcel = pd.DataFrame(self.synapticWeights)
        toExcel.to_excel(r'Logs\Weights'+str(self.layeridname)+'.xlsx')

# Encapsulates the Layers, Data Loading and Partitioning and the Activation Functions and Forward/Back Prop Algorithms

class NeuralNetwork():

    def __init__(self, layer1, layer2, layer3, activation, learningRate, iterationNo, useLog, usePlot):

        # ------DEFINE THE INPUT DATA SET TO USE WITH THE NETWORK-----------------------------------------------
        fileloc = r'PreparedFeatures\AllFeaturesNOPEAKS.xlsx'
        picWristOutput = r'PreparedFeatures\PickleWrist.pickle'
        picWristLabel = r'PreparedFeatures\PickleWristLabel.pickle'
        picWaistOutput = r'PreparedFeatures\PickleWaist.pickle'
        picWaistLabel = r'PreparedFeatures\PickleWaistLabel.pickle'
        np.set_printoptions()
        np.set_printoptions(formatter={"float_kind": "{:f}".format})

        #self.dataWaist = pd.read_excel(fileloc, sheet_name ='Waist', na_values=['NA'], usecols = 'A:CH')
        #self.waistLabel = pd.read_excel(fileloc, sheet_name ='Waist', na_values=['NA'], usecols = 'CI')

        # Import data from the Excel Spreadsheet of Feature Data
        try:
            self.dataWrist = pic.load(open(picWristOutput, 'rb'))
        except (EOFError, OSError, IOError, FileNotFoundError):
            self.dataWrist = pd.read_excel(fileloc, sheet_name='Wrist', na_values=['NA'], usecols='A:BP')
            pic.dump(self.dataWrist, open(picWristOutput, 'wb'))
        
        try:
            self.wristLabel = pic.load(open(picWristLabel, 'rb'))
        except (EOFError, OSError, IOError, FileNotFoundError):
            self.wristLabel = pd.read_excel(fileloc, sheet_name='Wrist', na_values=['NA'], usecols='BQ')
            pic.dump(self.wristLabel, open(picWristLabel, 'wb'))
    
        #with open(picWaistOutput, 'wb') as pickle_file:
            #pic.dump(self.dataWaist, pickle_file)
        #with open(picWaistLabel, 'wb') as pickle_file:
           # pic.dump(self.WaistLabel, pickle_file)

        

        # ------DEFINE THE DATA/ALGORITHMS AND PREFERENCES FROM THE PARAMETERS------------------------------------

        # Set the Two layers to the instances of NeuronLayer
        self.obj_layer1 = layer1
        self.obj_layer2 = layer2
        self.obj_layer3 = layer3
        # Define the Method of Activation
        self.activation = activation
        # Define the Learning Rate
        self.learningRate = learningRate
        # Define the Iteration Numbers
        self.Iterations = iterationNo
        # Define if feed forward is training or testing
        self.feedType = 'Training'
        
        # --------BASED ON THE PARAMETERS SELECT THE CORRECT FUNCTIONS TO USE IN FORWARD/BACK PROP---------------------

        # SIGMOID MAPPED TO FUNCTIONS
        if self.activation == 'Sigmoid':
            self.act_l1 = self.sigmoidL1
            self.act_l2 = self.sigmoidL2
            self.act_l3 = self.sigmoidL3
            self.back_l1 = self.sigmoidDerivativeL1
            self.back_l2 = self.sigmoidDerivativeL2
            self.back_l3 = self.sigmoidDerivativeL3
            istrue = True
        # RELU MAPPED TO FUNCTIONS
        elif self.activation == 'ReLu':
            self.act_l1 = self.reLuL1
            self.act_l2 = self.reLuL2
            self.act_l3 = self.reLuL3
            self.back_l1 = self.reLuDerL1
            self.back_l2 = self.reLuDerL2
            self.back_l3 = self.reLuDerL3
        # TODO: MAP TANH
        elif self.activation == 'tanh':
            #self.act_l1 = self.tanh
            #self.act_l2 = self.tanh
            #self.back_l1 = self.tanhDerivative
            #self.back_l2 = self.tanhDerivative
            istrue = True
       
        # Setup the Logging and Plotting Preferences
        self.logoutput = useLog
        self.plotoutput = usePlot

        #Define all Storage Values required
        self.layer1input = None
        self.layer2input = None
        self.layer3input = None

        #Output = Same size as corresponding input
        self.layer1Output = None
        self.layer2Output = None
        self.layer3Output = None

        #Back Prop output = Same Size as corresponding fwd output
        self.layer1back = None
        self.layer2back = None
        self.layer3back = None

        #Delta Arrays 
        self.layer1Delta = None
        self.layer2Delta = None
        self.layer3Delta = None

        #Error Arrays
        self.layer1Error = None
        self.layer2Error = None
        self.layer3Error = None

        #Loss Array
        self.loss = np.zeros(self.Iterations)

    def partitionData(self):
        # partition data sets into training and testing inputs and outputs - fixed size partition split
        self.x_trainWrist, self.x_testWrist, self.y_trainWrist, self.y_testWrist = train_test_split(
            self.dataWrist, self.wristLabel, shuffle=False, test_size=0.318)

        # For each value convert the data using the to_numpy function
        self.x_trainWrist = self.x_trainWrist.to_numpy()
        self.y_trainWrist = self.y_trainWrist.to_numpy()
        self.x_testWrist = self.x_testWrist.to_numpy()
        self.y_testWrist = self.y_testWrist.to_numpy()

        self.input = self.x_trainWrist

        # Initialise the Loop Counter to -1 so we can record the current cycle number
        self.iterationno = -1

        # print(self.x_trainWaist)
        # print(self.y_trainWaist)
        # print(self.x_testWaist)
        # print(self.y_testWaist)
        #print(self.x_trainWrist)
       # print(self.y_trainWrist)
        #print(self.x_testWrist)
        #print(self.y_testWrist)

    def allRowsDotProduct(self,inputmatrix,weightmatrix, bias):
        resultmatrix = np.zeros((inputmatrix.shape[0],weightmatrix.shape[1]))
        norows = inputmatrix.shape[0]
        i = 0
        while i < norows: 
            inputvect = inputmatrix[i]
            resultmatrix[i] = self.SynapseDotProduct(inputvect,weightmatrix, bias) 
            i +=1
        return resultmatrix

    def SynapseDotProduct(self,inputvect,weightmatrix, bias):
        synapsecount    = weightmatrix.shape[1]
        dotvalues = np.zeros(synapsecount)
        
        i = 0
        while i < synapsecount: 
            weightvect = weightmatrix[:, i]
            dotvalues[i] = self.dotProduct(inputvect,weightvect)
            dotvalues[i] += bias[i]
            i += 1
        return dotvalues 

    def dotProduct(self,input,weights):
        sum = 0
        for i in range(len(input)):
               dot = input[i]*weights[i]
               sum += dot
        return sum       

    def backProp(self):

        self.layer1back = np.copy(self.layer1Output)
        self.layer1Error = np.copy(self.layer1Output)
        self.layer2back = np.copy(self.layer2Output)
        self.layer3back = np.copy(self.layer3Output)
        #t_layer1WeightAdj = np.copy(self.obj_layer1.synapticWeights)
        #t_layer2WeightAdj = np.copy(self.obj_layer2.synapticWeights)
         
        # Calculate the error for layer 2. (The difference between the desired and predicted output) TODO: PUT BRACKETS AROUND PRECEDENCE
        self.layer3Error = self.layer3Output - self.y_trainWrist
        self.back_l3()
        self.layer3Delta = self.layer3Error * self.layer3back

        if self.logoutput:
            # Output Error and Deltas to File
            toExcel = pd.DataFrame(self.layer3Error)
            toExcel.to_excel(
                r'Logs\layer3Error' + str(self.activation) + str(self.iterationno)+'.xlsx')
            toExcel = pd.DataFrame(self.layer3Delta)
            toExcel.to_excel(
                r'Logs\layer3Delta' + str(self.activation) + str(self.iterationno)+'.xlsx')

        self.layer2Error = self.allRowsDotProduct(self.layer3Delta, self.obj_layer3.synapticWeights.T, np.zeros(self.obj_layer3.synapticWeights.shape[0]))
        self.back_l2()
        self.layer2Delta = self.layer2Error * self.layer2back

        if self.logoutput:
            # Output Error and Deltas to File
            toExcel = pd.DataFrame(self.layer2Error)
            toExcel.to_excel(
                r'Logs\layer2Error' + str(self.activation) + str(self.iterationno)+'.xlsx')
            toExcel = pd.DataFrame(self.layer2Delta)
            toExcel.to_excel(
                r'Logs\layer2Delta' + str(self.activation) + str(self.iterationno)+'.xlsx')

        # Layer 1 error calcluation. Determine layer 1 contribution to layer 2 error by checking the weight.
        #for i in range(len(self.layer2Delta)):
        self.layer1Error = self.allRowsDotProduct(self.layer2Delta, self.obj_layer2.synapticWeights.T, np.zeros(self.obj_layer2.synapticWeights.shape[0]))
        self.back_l1()
        self.layer1Delta = self.layer1Error * self.layer1back

        if self.logoutput:
            toExcel = pd.DataFrame(self.layer1Error)
            toExcel.to_excel(
                r'Logs\layer1Error' + str(self.activation) + str(self.iterationno)+'.xlsx')
            toExcel = pd.DataFrame(self.layer1Delta)
            toExcel.to_excel(
                r'Logs\layer1Delta' + str(self.activation) + str(self.iterationno)+'.xlsx')

        
         # Calculate weight adjustment

        t_layer1WeightAdj = self.allRowsDotProduct(self.input.T, self.layer1Delta, np.zeros(self.layer1Delta.shape[1]))
 
        t_layer2WeightAdj = self.allRowsDotProduct(self.layer1Output.T, self.layer2Delta, np.zeros(self.layer2Delta.shape[1]))

        t_layer3WeightAdj = self.allRowsDotProduct(self.layer2Output.T, self.layer3Delta, np.zeros(self.layer3Delta.shape[1]))

        # Adjust the weights
        self.obj_layer1.synapticWeights += t_layer1WeightAdj * self.learningRate
        self.obj_layer2.synapticWeights += t_layer2WeightAdj * self.learningRate
        self.obj_layer3.synapticWeights += t_layer3WeightAdj * self.learningRate

        if self.logoutput:
            toExcel = pd.DataFrame(self.obj_layer1.synapticWeights)
            toExcel.to_excel(r'Logs\layer1UpdatedWeight' + str(self.iterationno)+'.xlsx')
            toExcel = pd.DataFrame(self.obj_layer2.synapticWeights)
            toExcel.to_excel(r'Logs\layer2UpdatedWeight' +  str(self.iterationno)+'.xlsx')
            toExcel = pd.DataFrame(self.obj_layer3.synapticWeights)
            toExcel.to_excel(r'Logs\layer3UpdatedWeight' + str(self.iterationno)+'.xlsx')

        #Adjust the bias
        #self.obj_layer1.bias += self.learningRate * self.layer1Delta
        #self.obj_layer2.bias += self.learningRate * self.layer2Delta
        #self.obj_layer3.bias += self.learningRate * self.layer3Delta  

        #if self.logoutput:
           # toExcel = pd.DataFrame(self.obj_layer1.bias)
           # toExcel.to_excel(r'Logs\layer1bias' + str(self.iterationno)+'.xlsx')
          # toExcel = pd.DataFrame(self.obj_layer2.bias)
          #  toExcel.to_excel(r'Logs\layer2Updatedbias' + str(self.iterationno)+'.xlsx')
         #  toExcel = pd.DataFrame(self.obj_layer3.bias)
          #  toExcel.to_excel(r'Logs\layer3Updatedbias' + str(self.iterationno)+'.xlsx')

        t_loss = np.mean(np.square(self.y_trainWrist - self.layer3Output))
        self.loss[self.iterationno] = t_loss

        if self.logoutput and self.iterationno == self.Iterations-1:
            toExcel = pd.DataFrame(self.loss)
            toExcel.to_excel(
                r'Logs\Loss' + str(self.activation) + str(self.feedType)+'.xlsx')

    def feedForward(self):

        #Existing Dot Calcualation
        #self.layer1input = np.dot(self.input, self.obj_layer1.synapticWeights)
        self.layer1input = self.allRowsDotProduct(self.input,self.obj_layer1.synapticWeights, self.obj_layer1.bias)
        self.layer1Output = np.copy(self.layer1input)
        self.act_l1()

        if self.logoutput:
            toExcel = pd.DataFrame(self.layer1input)
            toExcel.to_excel(
                r'Logs\layer1input' + str(self.activation) + str(self.feedType) + str(self.iterationno)+'.xlsx')
            toExcel = pd.DataFrame(self.layer1Output)
            toExcel.to_excel(
                r'Logs\layer1output' + str(self.activation) + str(self.feedType) + str(self.iterationno)+'.xlsx')

        self.layer2input = self.allRowsDotProduct(self.layer1Output, self.obj_layer2.synapticWeights, self.obj_layer2.bias)
        self.layer2Output = np.copy(self.layer2input)
        self.act_l2()

        if self.logoutput:
            toExcel = pd.DataFrame(self.layer2input)
            toExcel.to_excel(
                r'Logs\layer2input' + str(self.activation) + str(self.feedType) + str(self.iterationno)+'.xlsx')
            toExcel = pd.DataFrame(self.layer2Output)
            toExcel.to_excel(
                r'Logs\layer2output' + str(self.activation) + str(self.feedType) + str(self.iterationno)+'.xlsx')
        
        self.layer3input = self.allRowsDotProduct(self.layer2Output, self.obj_layer3.synapticWeights, self.obj_layer3.bias)
        self.layer3Output = np.copy(self.layer3input)
        self.act_l3()

        if self.logoutput:
            toExcel = pd.DataFrame(self.layer3input)
            toExcel.to_excel(
                r'Logs\layer3input' + str(self.activation) + str(self.feedType) + str(self.iterationno)+'.xlsx')
            toExcel = pd.DataFrame(self.layer3Output)
            toExcel.to_excel(
                r'Logs\layer3output' + str(self.activation) + str(self.feedType) + str(self.iterationno)+'.xlsx')

    def trainNetwork(self):

        for i in range(self.Iterations):

            # Store the Iteration Number
            self.iterationno = i

            # Pass training set through the neural network
            self.feedForward()

            # Trigger Backwards Propagation
            self.backProp()

            if self.plotoutput:
                plt.plot(self.layer1Delta)
                plt.show()
                plt.plot(self.layer2Delta)
                plt.show()


            tbreakval = 1

        # Adjust the bias
        # for num in layer1Delta:
        #self.layer1.bias += num * self.learningRate

        # for num in layer2Delta:
        #self.layer2.bias += num * self.learningRate

        print("Training Loop Complete")

    # -------------OO LAYER-BASED ACTIVATION FUNCTIONS (NO PARAMETERS - CLASS MEMBER BASED)-------------------------------------
    def sigmoidL1(self):
        # define the sigmoid activation function

        for i in range(len(self.layer1input)):
            for j in range(len(self.layer1input[i])):

                if self.layer1input[i][j] >= 0:
                    z = np.exp(-self.layer1input[i][j])
                    self.layer1Output[i][j] = 1. / (1. + z)
                elif self.layer1input[i][j] < 0:
                    z = np.exp(self.layer1input[i][j])
                    self.layer1Output[i][j] = z / (1. + z)

    def sigmoidL2(self):
        # define the sigmoid activation function

        for i in range(len(self.layer2input)):
            for j in range(len(self.layer2input[i])):

                if self.layer2input[i][j] >= 0:
                    z = np.exp(-self.layer2input[i][j])
                    self.layer2Output[i][j] = 1. / (1. + z)
                elif self.layer2input[i][j] < 0:
                    z = np.exp(self.layer2input[i][j])
                    self.layer2Output[i][j] = z / (1. + z)
    
    def sigmoidL3(self):
        # define the sigmoid activation function

        for i in range(len(self.layer3input)):
            for j in range(len(self.layer3input[i])):

                if self.layer3input[i][j] >= 0:
                    z = np.exp(-self.layer3input[i][j])
                    self.layer3Output[i][j] = 1. / (1. + z)
                elif self.layer3input[i][j] < 0:
                    z = np.exp(self.layer3input[i][j])
                    self.layer3Output[i][j] = z / (1. + z)

    def sigmoidDerivativeL1(self):
        # calculate the derivative of sigmoid
        self.layer1back = self.layer1Output * (1 - self.layer1Output)
    
    def sigmoidDerivativeL2(self):
        # calculate the derivative of sigmoid
        self.layer2back = self.layer2Output * (1 - self.layer2Output)
    
    def sigmoidDerivativeL3(self):
        # calculate the derivative of sigmoid
        self.layer3back = self.layer3Output * (1 - self.layer3Output)

    def reLuL1(self):
        for i in range(len(self.layer1input)):
            for j in range(len(self.layer1input[i])):
                if self.layer1input[i][j] > 0:
                    self.layer1Output[i][j] = self.layer1input[i][j]
                else:
                    self.layer1Output[i][j] = 0

    def reLuL2(self):
        for i in range(len(self.layer2input)):
            for j in range(len(self.layer2input[i])):
                if self.layer2input[i][j] > 0:
                    self.layer2Output[i][j] = self.layer2input[i][j]
                else:
                    self.layer2Output[i][j] = 0

    def reLuL3(self):
        for i in range(len(self.layer3input)):
            for j in range(len(self.layer3input[i])):
                if self.layer3input[i][j] > 0:
                    self.layer3Output[i][j] = self.layer3input[i][j]
                else:
                    self.layer3Output[i][j] = 0

    def reLuDerL1(self):
        for i in range(len(self.layer1Output)):
            for j in range(len(self.layer1Output[i])):

                if self.layer1Output[i][j] <= 0:
                    self.layer1back[i][j] = 0

                elif self.layer1Output[i][j] > 0:
                    self.layer1back[i][j] = 1

    def reLuDerL2(self):
        for i in range(len(self.layer2Output)):
            for j in range(len(self.layer2Output[i])):

                if self.layer2Output[i][j] <= 0:
                    self.layer2back[i][j] = 0

                elif self.layer2Output[i][j] > 0:
                    self.layer2back[i][j] = 1

    def reLuDerL3(self):
        for i in range(len(self.layer3Output)):
            for j in range(len(self.layer3Output[i])):

                if self.layer3Output[i][j] <= 0:
                    self.layer3back[i][j] = 0

                elif self.layer3Output[i][j] > 0:
                    self.layer3back[i][j] = 1

# ---------------MAIN EXECUTION FUNCTION-------------------------------------------------------
if __name__ == "__main__":

    # initalize the neurons
    layer1 = NeuronLayer(68, 30, "Input_to_Hidden_1")
    layer2 = NeuronLayer(30, 18, "Hidden_1_to_Hidden_2")
    layer3 = NeuronLayer(18, 1, "Hidden_2_to_Output")

    # Select the Logging
    uselog = True
    useplot = False

    # initializing the neuron class
    neural_network = NeuralNetwork(
        layer1, layer2, layer3, 'Sigmoid', 0.00001, 10, uselog, useplot)

    # Partition the data sets
    neural_network.partitionData()
    data = pd.DataFrame(neural_network.y_trainWrist)
    data.to_excel(r'Logs\Data.xlsx')

    neural_network.trainNetwork()
    loss = pd.DataFrame(neural_network.loss)

    neural_network.input = neural_network.x_testWrist
    neural_network.feedType = 'Test'
    neural_network.feedForward()

    print(neural_network.layer3Output)
    #print(accuracy_score(neural_network.y_testWrist, output))
    loss = np.mean(np.square(neural_network.y_testWrist - neural_network.layer3Output))
    print(loss)
    #loss = pd.DataFrame(loss)
    #loss.to_excel(r'Logs\loss' + str(neural_network.activation) + str(neural_network.feedType) +'.xlsx')

    output = pd.DataFrame(neural_network.layer3Output)
    output.to_excel(r'Logs\Results' + str(neural_network.activation) + str(neural_network.feedType) + str(neural_network.iterationno)+'.xlsx')
