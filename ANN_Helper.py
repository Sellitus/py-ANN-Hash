
import hashlib
import random
import string




# Generate test and training dataset
def generate_dataset(numData, stringLength, hashType):
    X = []
    Y = []

    for i in range(numData):
        # First generate a random string of length stringLength
        newString = ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
                            for _ in range(stringLength))

        # Next hash the string
        if(hashType == "sha1"):
            newHash = hashlib.sha1(newString.encode('utf-8')).hexdigest()
            hashLength = 160
        elif(hashType == "md5"):
            newHash = hashlib.md5(newString.encode('utf-8')).hexdigest()
            hashLength = 128

        # Convert newString into inputs that the ANN understands
        convertedString = []
        for currLetter in list(newString):
            # Convert currLetter to its int representation
            currLetter = ord(currLetter)

            # Scale it to between 0 and 1
            #currLetter = currLetter / 128;

            # Append it to the convertedString array
            convertedString.append(currLetter)

        # Convert the newHash into outputs the ANN understands

        # Convert the convertedHash to a binary string
        convertedHash = bin(int(newHash, 16))[2:]

        # Make sure the length is 128 (0s will be stripped from front)
        while (len(convertedHash) < hashLength):
            convertedHash = '0' + convertedHash

        # Convert the binary string into a binary array
        convertedHash = list(map(float, convertedHash))

        # Append the input and outputs to the X and Y arrays
        X.append(convertedString)
        Y.append(convertedHash)

    return [X, Y]



# Return the difference in bits between the two bitstrings
def calculate_cost(X, Y):
    numWrongBits = 0

    for i in range(X):
        if X[i] != Y[i]:
            numWrongBits += 1

    return numWrongBits



def start_evaluation(model, X, Y):
    # Evaluate the model
    scores = model.evaluate(X, Y)
    return scores


