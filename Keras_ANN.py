
from ANN_Helper import *



# Create first network with Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

import numpy as np

from threading import Thread


hashType = "md5"
numHiddenLayers = 20
numNodes = 512
numEpoch = 9999999
sizeDatasets = 10000
sizeInputString = 6
batchSize = 1000


# Set hashSize based on hash type
if (hashType == "sha1"):
    hashSize = 160
elif (hashType == "md5"):
    hashSize = 128


# fix random seed for reproducibility
seed = 7
random.seed(seed)

# load training datasets
X, Y = generate_dataset(sizeDatasets, sizeInputString, hashType, )


# create model
model = Sequential()



# Set activation functions and save length to lenActFunc
activation_functions = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
lenActFunc = len(activation_functions)

init_functions = ['uniform', 'normal']
lenInitFunc = len(init_functions)



# Input layer
model.add(Dense(numNodes, input_dim=sizeInputString, init='uniform', activation='relu'))

'''
# Add however many hidden layers specified in numHiddenLayers
for _ in range(numHiddenLayers):
    random_activation = activation_functions[random.randrange(0,lenActFunc,1)]

    model.add(Dense(numNodes, init='uniform', activation=random_activation))

    # May as well add another relu layer
    model.add(Dense(numNodes, init='uniform', activation='relu'))

# Output layer
model.add(Dense(hashSize, init='uniform', activation='sigmoid'))
# Add advanced layer
act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
model.add(act)
'''


for _ in range(numHiddenLayers):
    random_activation = activation_functions[random.randrange(0, lenActFunc, 1)]
    #random_init = init_functions[random.randrange(0, lenInitFunc, 1)]
    model.add(Dense(numNodes, init='uniform', activation=random_activation))




model.add(Dense(hashSize, init='uniform', activation='sigmoid'))

act = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
model.add(act)



# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=numEpoch, batch_size=batchSize)

# Evaluate the model
scores = start_evaluation(model, X, Y)

#thread = Thread(target = start_evaluation, args = (model, X, Y, ))
#thread.start()
#thread.join()


# Generate test datasets
testX, testY = generate_dataset(sizeDatasets, sizeInputString, hashType)


predictions = model.predict(testX)
print()

print(len(testY))

print("Expected Output: " + str(testY[1]))
print("Predicted Output: " + str(predictions[1]))

actual_error = sum(sum(abs(testY - predictions))) / sizeDatasets / hashSize * 100
print("Absolute Error: " + str(actual_error) + "%")

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




