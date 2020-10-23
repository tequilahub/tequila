Exploring the application of quantum circuits in convolutional neural networks
==============================================================================

This tutorial will guide you through implementing a hybrid
quantum-classical convolutional neural network using Tequila along with
other packages such as Tensorflow. We will then train the model on the
MNIST dataset, which contains images of handwritten numbers classifed
according to the digit they represent. Finally, we will compare the
accuracy and loss of models with and without the quantum preprocessing.

Inspriation for this tutorial comes from `Pennylane: Quanvolutional
Neural
Networks <https://pennylane.ai/qml/demos/tutorial_quanvolution.html>`__.
We will similarly follow the method proposed in the reference paper used
for this tutorial, `Henderson at al
(2020) <https://doi.org/10.1007/s42484-020-00012-y>`__.

Background
----------

Convolutional Neural Nets
^^^^^^^^^^^^^^^^^^^^^^^^^

An excellent high-level explanation of convolutional neural networks can
be found `here <https://www.youtube.com/watch?v=FmpDIaiMIeA>`__.
Alternatively, an excellent written explanation can be found
`here <http://neuralnetworksanddeeplearning.com/chap6.html>`__ and for
more information, the wikipedia article can be found
`here <https://en.wikipedia.org/wiki/Convolutional_neural_network>`__.

In summary, a convolutional neural network includes preprocessing layers
prior to optimisation layers so that features in the input (which are
often images) are extracted and amplified. The result is a model with
greater predictive power. This processing also improves classification
of images as it extracts features even if they are translocated between
images. This means that searching for a particular pixel distribution
(for example the shape of a curve or line may be a useful feature when
classifying digits) is not dependant on the distribution being in an
identical location in each image where it is present. The convolutional
process extracts this information even if it is slightly rotated or
translocated.

The implementation of the convolutional layer involves a grid for each
feature being passed over the entire image. At each location, a score is
calculated representing how well the feature and the section of the
image match, and this becomes the value of the corresponding pixel in
the output image. As a guide, a large score represents a close match,
generally meaning that the feature is present at that location of the
image, and a low score represents the absence of a match.

Our Approach
^^^^^^^^^^^^

Our general approach is similar to that used in a conventional
convolutional neural network however the initial processing occurs by
running the images through a quantum circuit instead of a convolutional
filter. Each simulation of a circuit represents one 3x3 filter being
applied to one 3x3 region of one image. The construction of the circuit
is randomised (see below), however this construction only occurs once
per filter such that each region of the image being transformed by the
same filter gets run through the same circuit. A single, scalar output
is generated from the circuit which is used as the pixel strength of the
output image, and the remainder of the neural net uses only classical
processing, specifically two further convolutional layers, max pooling
and two fully connected layers. This architecture has been chosen to
closely mimic the structure used in our reference paper (Henderson et
al, 2020), however as they note, "The QNN topology chosen in this work
is not fixed by nature ... the QNN framework was designed to give users
complete control over the number and order of quanvolutional layers in
the architecture. The topology explored in this work was chosen because
it was the simplest QNN architecture to use as a baseline for comparison
against other purely classical networks. Future work would focus on
exploring the impact of more complex architectural variations."



Quantum Processing
^^^^^^^^^^^^^^^^^^

Henderson et al summarise the use of quantum circuits as convolutional
layers: "Quanvolutional layers are made up of a group of N quantum
filters which operate much like their classical convolutional layer
counterparts, producing feature maps by locally transforming input data.
The key difference is that quanvolutional filters extract features from
input data by transforming spatially local subsections of data using
quantum circuits." Our approach to the circuit design is based on the
paper and is as follows:

1) The input images are iterated over and each 3x3 region is embedded
   into the quantum circuit using the threshold function:

   .. math::

      |\psi \rangle = \begin{cases} 
                        |0\rangle & if & strength\leq 0 \\
                        |1\rangle & if & strength > 0
                     \end{cases}

As the pixel strengths are normalised to values between -0.5 and 0.5, it
is expected that brighter regions of the image will intialise their
corresponding qubit in the state :math:`|1\rangle` and darker regions
will intitialise the state :math:`|0\rangle`. Each pixel is represented
by one qubit, such that 9 qubits are used in total, and this quantum
circuit is reused for each 3x3 square in the filter.

2) We next apply a random circuit to the qubits. To implement this, a
   random choice from Rx, Ry and Rz gates is applied to a random qubit,
   and the total number of gates applied in each layer is equal to the
   number of qubits. With a set probability (which we set to 0.3), a
   CNOT gate will be applied instead of the rotation to two random
   qubits. We have chosen to set the parameters of rotation with random
   numbers between (0,2π) however futher optimisation of the model could
   be found from using a variational circuit and optimising these
   parameters.

3) Further layers could be applied of the random gates. To simplify, we
   only apply one layer.

4) A scalar is outputted from the circuit and used as the corresponding
   pixel in the output image. We generate this number using the
   following method. The state vector of the final state of the circuit
   is simulated and the state corresponding to the most likely output
   (largest modulus) is selected. We then calculate the number of qubits
   for this state which are measured as a :math:`|1\rangle`.

5) A total of four filters are applied to each image, and for each
   filter steps 1-3 are repeated with a different randomised circuit.
   The output image therefore contains a third dimension with four
   channels representing the four different outputted values which each
   filters produced.



Code and Running the Program
----------------------------

The following code cell is used to import the necessary packages and to
set parameters.

.. code:: ipython3

    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import tequila as tq
    
    from operator import itemgetter
    from tensorflow import keras
    
    n_filters = 4               # Number of convolutional filters
    filter_size = 3             # Size of filter = nxn (here 3x3)
    pool_size = 2               # Used for the pooling layer
    n_qubits = filter_size ** 2 # Number of qubits
    n_layers = 1                # Number of quantum circuit layers
    n_train = 1000              # Size of the training dataset
    n_test = 200                # Size of the testing dataset
    n_epochs = 100              # Number of optimization epochs
    
    SAVE_PATH = "quanvolution/" # Data saving folder
    PREPROCESS = False          # If False, skip quantum processing and load data from SAVE_PATH
    tf.random.set_seed(1)       # Seed for TensorFlow random number generator

We start by creating the Dataset class. Here, we load the images and
labels of handwritten digits from the MNIST dataset. We then reduce the
number of images from 60,000 and 10,000 (for the training and testing
sets respectively) down to the variables n\_train and n\_test, normalise
the pixel values to within the range (-0.5,0.5) and reshape the images
by adding a third dimension. Each image's shape is therefore transformed
from (28, 28) to (28, 28, 1) as this is necessary for the convolutional
layer.

.. code:: ipython3

    class Dataset:
    
        def __init__(self):
            # Loading the full dataset of images from keras
            # Shape of self.train_images is (60000, 28, 28), shape of self.train_labels is (60000,)
            # For self.test_images and self.test_labels, shapes are (10000, 28, 28) and (10000,)
            mnist_dataset = keras.datasets.mnist
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist_dataset.load_data()
    
            # Reduce dataset size to n_train and n_test
            # First dimension of shapes are reduced to n_train and n_test
            self.train_images = self.train_images[:n_train]
            self.train_labels = self.train_labels[:n_train]
            self.test_images = self.test_images[:n_test]
            self.test_labels = self.test_labels[:n_test]
    
            # Normalize pixel values within -0.5 and +0.5
            self.train_images = (self.train_images / 255) - 0.5
            self.test_images = (self.test_images / 255) - 0.5
    
            # Add extra dimension for convolution channels
            self.train_images = self.train_images[..., tf.newaxis]
            self.test_images = self.test_images[..., tf.newaxis]

The next code cell contains the class used to generate the quantum
circuit. In theory, the circuit could be either structured or random. We
form a randomised circuit to match the reference paper (Henderson et al,
2020), however for simplicity, our implementation differs in some ways.
We choose to use only use single qubit Rx(\ :math:`\theta`),
Ry(\ :math:`\theta`) and Rz(\ :math:`\theta`) gates and the two qubit
CNOT gate compared to the choice of single qubit X(\ :math:`\theta`),
Y(\ :math:`\theta`), Z(\ :math:`\theta`), U(\ :math:`\theta`), P, T, H
and two qubit CNOT, SWAP, SQRTSWAP, or CU gates used in the paper.
Furthermore, we chose to assign a two qubit gate to any random qubits
with a certain probability (labelled ratio\_imprim, set to 0.3) rather
than setting a connection probabiltiy between each pair of qubits (this
approach follows the Pennylane tutorial). The seed is used for
reproducability and its value is set depending on which filter the
circuit represents (see QuantumModel below).

The parameters used for the rotation gates have the potential to be
optimised using a cost function. For simplicity, and to mirror the
paper, here we will use random parameters and we will not include these
in the optimisation of the model. This means that the quantum processing
only needs to happen once, prior to creating the neural net.

.. code:: ipython3

    class QuantumCircuit:
        
        def __init__(self, seed=None):
            # Set random seed for reproducability
            if seed: np.random.seed(seed)
            
            # Encode classical information into quantum circuit
            # Bit flip gate is applied if the pixel strength > 0
            self.circ = tq.QCircuit()
            for i in range(n_qubits):
                self.circ += tq.gates.X(i, power='input_{}'.format(i))
    
            # Add random layers to the circuit
            self.circ += self.random_layers()
        
        def random_layers(self, ratio_imprim=0.3):
            # Initialise circuit
            circuit = tq.QCircuit()
    
            # Iterate over the number of layers, adding rotational and CNOT gates
            # The number of rotational gates added per layer is equal to the number of qubits in the circuit
            for i in range(n_layers):
                j = 0
                while (j < n_qubits):
                    if np.random.random() > ratio_imprim:
                        # Applies a random rotation gate to a random qubit with probability (1 - ratio_imprim)
                        rnd_qubit = np.random.randint(n_qubits)
                        circuit += np.random.choice(
                            [tq.gates.Rx(angle='l_{},th_{}'.format(i,j), target=rnd_qubit),
                             tq.gates.Ry(angle='l_{},th_{}'.format(i,j), target=rnd_qubit),
                             tq.gates.Rz(angle='l_{},th_{}'.format(i,j), target=rnd_qubit)])
                        j += 1
                    else:
                        # Applies the CNOT gate to 2 random qubits with probability ratio_imprim
                        if n_qubits > 1:
                            rnd_qubits = np.random.choice(range(n_qubits), 2, replace=False)
                            circuit += tq.gates.CNOT(target=rnd_qubits[0], control=rnd_qubits[1])
            return circuit

As an example to show the circuit used in this program, an instance of a
circuit is drawn below. This will differ between calls if you remove the
seed variable due to the random nature of forming the circuit.

.. code:: ipython3

    circuit = QuantumCircuit(seed=2)
    tq.draw(circuit.circ, backend='qiskit')

We next show the QuantumModel class, used to generate the neural network
for the images which undergo pre-processing through the quantum
convolutional layer. If PREPROCESSING is set to True, each image from
the dataset undergoes processing through a number of quantum circuits,
determined by n\_filters. The embedding used, the structure of the
circuit and the method of extracting the output are described in the
background of this tutorial.

We use tensorflow to construct the neural net. The implementation we use
contains two conventional convolutional layers, each followed by max
pooling, and then one fully connected with 1024 nodes before the softmax
output layer. We use a Relu activation function for the convolutional
and fully connected layers. See the background section of this tutorial
for some context on this choice of neural net.

.. code:: ipython3

    class QuantumModel:
    
        def __init__(self, dataset, parameters):
            # Initialize dataset and parameters
            self.ds = dataset
            self.params = parameters
            
            # The images are run through the quantum convolutional layer
            self.convolutional_layer()
    
            # The model is initialized
            self.model = keras.models.Sequential([
                keras.layers.Conv2D(n_filters, filter_size, activation='relu'),
                keras.layers.MaxPooling2D(pool_size=pool_size),
                keras.layers.Conv2D(n_filters, filter_size, activation='relu'),
                keras.layers.MaxPooling2D(pool_size=pool_size),
                keras.layers.Flatten(),
                keras.layers.Dense(1024, activation="relu"),
                keras.layers.Dense(10, activation="softmax")
            ])
    
            # Compile model using the Adam optimiser
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
        
        def convolutional_layer(self):
            if PREPROCESS == True:
                # Initate arrays to store processed images
                self.q_train_images = [np.zeros((28-2, 28-2, n_filters)) for _ in range(len(self.ds.train_images))]
                self.q_test_images = [np.zeros((28-2, 28-2, n_filters)) for _ in range(len(self.ds.test_images))]
                
                # Loop over the number of filters, applying a different randomised quantum circuit for each
                for i in range(n_filters):
                    print('Filter {}/{}\n'.format(i+1, n_filters))
                    
                    # Construct circuit
                    # We set the seed to be i+1 so that the circuits are reproducable but the design differs between filters
                    # We use i+1 not i to avoid setting the seed as 0 which sometimes produces random behaviour
                    circuit = QuantumCircuit(seed=i+1)
                    
                    # Apply the quantum processing to the train_images, analogous to a convolutional layer
                    print("Quantum pre-processing of train images:")
                    for j, img in enumerate(self.ds.train_images):
                        print("{}/{}        ".format(j+1, n_train), end="\r")
                        self.q_train_images[j][...,i] = (self.filter_(img, circuit, self.params[i]))
                    print('\n')
    
                    # Similarly for the test_images
                    print("Quantum pre-processing of test images:")
                    for j, img in enumerate(self.ds.test_images):
                        print("{}/{}        ".format(j+1, n_test), end="\r")
                        self.q_test_images[j][...,i] = (self.filter_(img, circuit, self.params[i]))
                    print('\n')
    
                # Transform images to numpy array
                self.q_train_images = np.asarray(self.q_train_images)
                self.q_test_images = np.asarray(self.q_test_images)
                
                # Save pre-processed images
                np.save(SAVE_PATH + "q_train_images.npy", self.q_train_images)
                np.save(SAVE_PATH + "q_test_images.npy", self.q_test_images)
            
            # Load pre-processed images
            self.q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
            self.q_test_images = np.load(SAVE_PATH + "q_test_images.npy")
    
        def filter_(self, image, circuit, variables):
            # Initialize output image
            output = np.zeros((28-2, 28-2))
    
            # Loop over the image co-ordinates (i,j) using a 3x3 square filter
            for i in range(28-2):
                for j in range(28-2):
    
                    # Extract the value of each pixel in the 3x3 filter grid
                    image_pixels = [
                        image[i,j,0],
                        image[i,j+1,0],
                        image[i,j+2,0],
                        image[i+1,j,0],
                        image[i+1,j+1,0],
                        image[i+1,j+2,0],
                        image[i+2,j,0],
                        image[i+2,j+1,0],
                        image[i+2,j+2,0]
                    ]
    
                    # Construct parameters used to embed the pixel strength into the circuit
                    input_variables = {}
                    for idx, strength in enumerate(image_pixels):
                        # If strength > 0, the power of the bit flip gate is 1
                        # Therefore this qubit starts in state |1>
                        if strength > 0:
                            input_variables['input_{}'.format(idx)] = 1
                        # Otherwise the gate is not applied and the initial state is |0>
                        else:
                            input_variables['input_{}'.format(idx)] = 0
    
                    # Find the statevector of the circuit and determine the state which is most likely to be measured
                    wavefunction = tq.simulate(circuit.circ, variables={**variables, **input_variables})
                    amplitudes = [(k,(abs(wavefunction(k)))) for k in range(2**n_qubits) if wavefunction(k)]
                    max_idx = max(amplitudes,key=itemgetter(1))[0]
                    
                    # Count the number of qubits which output '1' in this state
                    result = len([k for k in str(bin(max_idx))[2::] if k == '1'])
                    output[i,j] = result
            return output
    
        def train(self):
            # Train the model on the dataset
            self.history = self.model.fit(
                self.q_train_images,
                self.ds.train_labels,
                validation_data=(self.q_test_images, self.ds.test_labels),
                batch_size=4,
                epochs=n_epochs,
                verbose=2
            )

We also create a ClassicalModel class to run the images through a
conventional convolutional neural network. The design of the neural net
used here is identical to the QuantumModel class, however the images
used are directly from the dataset and therefore have not been processed
through the quantum layer. We include this as a control to compare the
results from the quantum model.

.. code:: ipython3

    class ClassicalModel:
    
        def __init__(self, dataset):
            # Initialize dataset and parameters
            self.ds = dataset
    
            # The model is initialized
            self.model = keras.models.Sequential([
                keras.layers.Conv2D(n_filters, filter_size, activation='relu'),
                keras.layers.MaxPooling2D(pool_size=pool_size),
                keras.layers.Conv2D(n_filters, filter_size, activation='relu'),
                keras.layers.MaxPooling2D(pool_size=pool_size),
                keras.layers.Flatten(),
                keras.layers.Dense(1024, activation="relu"),
                keras.layers.Dense(10, activation="softmax")
            ])
    
            # Compile model using the Adam optimiser
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.00005),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
        
        def train(self):
            # Train the model on the dataset
            self.history = self.model.fit(
                self.ds.train_images,
                self.ds.train_labels,
                validation_data=(self.ds.test_images, self.ds.test_labels),
                batch_size=4,
                epochs=n_epochs,
                verbose=2
            )

We are now able to run our program! The following code does this using
the quantum\_model and classical\_model functions. Although the
implementations are similar, quantum\_model additionally defines the
parameters used for the rotational gates in the circuit. We have limited
the value of each parameter to the range (0,2π).

Running the program takes some time. Our results are plotted below, so
if you would rather not wait, either reduce the numbers in n\_train and
n\_test or skip ahead!

.. code:: ipython3

    def quantum_model():
        # Generating parameters, each maps to a random number between 0 and 2*π
        # parameters is a list of dictionaries, where each dictionary represents the parameter
        # mapping for one filter
        parameters = []
        for i in range(n_filters):
            filter_params = {}
            for j in range(n_layers):
                for k in range(n_qubits):
                    filter_params[tq.Variable(name='l_{},th_{}'.format(j,k))] = np.random.uniform(high=2*np.pi)
            parameters.append(filter_params)
            
        # Initalise the dataset
        ds = Dataset()
        
        # Initialise and train the model
        model = QuantumModel(ds, parameters=parameters)
        model.train()
        
        # Store the loss and accuracy of the model to return
        loss = model.history.history['val_loss']
        accuracy = model.history.history['val_accuracy']
    
        return model
    
    def classical_model():
        # Initialise the dataset
        ds = Dataset()
        
        # Initialise and train the model
        model = ClassicalModel(ds)
        model.train()
        
        # Store the loss and accuracy of the model to return
        loss = model.history.history['val_loss']
        accuracy = model.history.history['val_accuracy']
        
        return model
    
    model_q = quantum_model()
    model_c = classical_model()

Plotting the Results
--------------------

The graphs showing the accuracy and loss of our models are included in
this text box. These were generated using the function plot, available
below. As shown, the results from the quantum processing lead to a model
comparable to the classical control in both accuracy and loss. After
running for 100 epochs, the quantum model results in a validation set
accuracy of 0.9350, compared to the fully classical model which has a
validation set accuracy of 0.9150.

.. code:: ipython3

    def plot(model_q, model_c):
    
        plt.style.use("seaborn")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))
    
        # Plotting the graph for accuracy
        ax1.plot(model_q.history.history['val_accuracy'], color="tab:red", label="Quantum")
        ax1.plot(model_c.history.history['val_accuracy'], color="tab:green", label="Classical")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim([0,1])
        ax1.set_xlabel("Epoch")
        ax1.legend()
    
        # Plotting the graph for loss
        ax2.plot(model_q.history.history['val_loss'], color="tab:red", label="Quantum")
        ax2.plot(model_c.history.history['val_loss'], color="tab:green", label="Classical")
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    plot(model_q, model_c)

Evaluating the Model
--------------------

Let us now compare the behaviour of the two models. We do this by
running the test images through each with the optimised weights and
biases and seeing the results of the classification. This process is
implemented using the Classification class, shown below.

Overall, our quantum model misclassified images 34, 37, 42, 54, 67, 74,
120, 127, 143, 150, 152, 166, and 185. The classical model misclassified
images 8, 16, 21, 23, 54, 60, 61, 67, 74, 93, 113, 125, 134, 160, 168,
178, and 196. This means that in total, the quantum model misclassified
13 images and the classical model misclassified 17 images. Of these,
only images 54, 67, and 74 were misclassified by both.

.. code:: ipython3

    from termcolor import colored
    
    class Classification:
        
        def __init__(self, model, test_images):
            # Initialising parameters
            self.model = model
            self.test_images = test_images
            self.test_labels = model.ds.test_labels
    
        def classify(self):
            # Create predictions on the test set
            self.predictions = np.argmax(self.model.model.predict(self.test_images), axis=-1)
    
            # Keep track of the indices of images which were classified correctly and incorrectly
            self.correct_indices = np.nonzero(self.predictions == self.test_labels)[0]
            self.incorrect_indices = np.nonzero(self.predictions != self.test_labels)[0]
        
        def print_(self):
            # Printing the total number of correctly and incorrectly classified images
            print(len(self.correct_indices)," classified correctly")
            print(len(self.incorrect_indices)," classified incorrectly")
            print('\n')
    
            # Printing the classification of each image
            for i in range(n_test):
                print("Image {}/{}".format(i+1, n_test))
                if i in self.correct_indices:
                    # The image was correctly classified
                    print('model predicts: {} - true classification: {}'.format(
                        self.predictions[i], self.test_labels[i]))
                else:
                    # The image was not classified correctly
                    print(colored('model predicts: {} - true classification: {}'.format(
                        self.predictions[i], self.test_labels[i]), 'red'))

.. code:: ipython3

    print('Quantum Model')
    q_class = Classification(model_q, model_q.q_test_images)
    q_class.classify()
    q_class.print_()
    
    print('\n')
    
    print('Classical Model')
    c_class = Classification(model_c, model_c.ds.test_images)
    c_class.classify()
    c_class.print_()

Lastly, we can see the effect that the quantum convolutional layer
actually has on the images by plotting images after they have been run
through the quantum filters, and to do this we use the function
visualise, shown below. Included in this text box is a plot showing four
images which have been run through our filters. The top row shows images
from the original dataset, and each subsequent row shows the result from
each of the four filters on that original image. It can be seen that the
processing preserves the global shape of the digit while introducing
local distortion.

.. code:: ipython3

    def visualise(model):
        # Setting n_samples to be the number of images to print
        n_samples = 4
        
        fig, axes = plt.subplots(1 + n_filters, n_samples, figsize=(10, 10))
        
        # Iterate over each image
        for i in range(n_samples):
            
            # Plot the original image from the dataset
            axes[0, 0].set_ylabel("Input")
            if i != 0:
                axes[0, i].yaxis.set_visible(False)
            axes[0, i].imshow(model.ds.train_images[i, :, :, 0], cmap="gray")
    
            # Plot the images generated by each filter
            for c in range(n_filters):
                axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
                if i != 0:
                    axes[c, i].yaxis.set_visible(False)
                axes[c + 1, i].imshow(model.q_train_images[i, :, :, c], cmap="gray")
    
        plt.tight_layout()
        plt.show()
        
    visualise(model_q)

Resources used to make this tutorial:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. `Pennylane: Quanvolutional Neural
   Networks <https://pennylane.ai/qml/demos/tutorial_quanvolution.html>`__
2. Henderson, M., Shakya, S., Pradhan, S. et al. Quanvolutional neural
   networks: powering image recognition with quantum circuits. Quantum
   Mach. Intell. 2, 1–9 (2020).
   https://doi.org/10.1007/s42484-020-00012-y
3. `Keras for Beginners: Implementing a Convolutional Neural Network.
   Victor Zhou <https://victorzhou.com/blog/keras-cnn-tutorial/>`__.
4. `CNNs, Part 1: An Introduction to Convolutional Neural Networks.
   Victor Zhou <https://victorzhou.com/blog/intro-to-cnns-part-1/>`__.
5. `How Convolutional Neural Networks
   work <https://www.youtube.com/watch?v=FmpDIaiMIeA>`__
6. `Neural Networks and Deep Learning, chapter 6. Michael
   Nielsen <http://neuralnetworksanddeeplearning.com/chap6.html>`__
