import numpy as np
from sklearn import preprocessing

# Example labels
labels = np.array([1,5,3,2,1,4,2,1,3])

# Create the encoder
lb = preprocessing.LabelBinarizer()

# Here the encoder finds the classes and assigns one-hot vectors
lb.fit(labels)

# And finally, transform the labels into one-hot encoded vectors
output = lb.transform(labels)

print(output)
# Should output:
# >>> array([[1, 0, 0, 0, 0],
#            [0, 0, 0, 0, 1],
#            [0, 0, 1, 0, 0],
#            [0, 1, 0, 0, 0],
#            [1, 0, 0, 0, 0],
#            [0, 0, 0, 1, 0],
#            [0, 1, 0, 0, 0],
#            [1, 0, 0, 0, 0],
#            [0, 0, 1, 0, 0]])
