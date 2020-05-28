import numpy as np
from qp import solve_QP

def linear_kernel(xj, xk):
    """
    Kernel Function, linear kernel (ie: regular dot product)

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :return: float64
    """
    #TODO. DONE
    xj_transpose = np.transpose(xj)
    return np.matmul(xj_transpose, xk)

def rbf_kernel(xj, xk, gamma = .1):
    """
    Kernel Function, radial basis function kernel or gaussian kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param gamma: parameter of the RBF kernel.
    :return: float64
    """
    # TODO. DONE
    difference_array = np.subtract (xj, xk)
    norm = np.linalg.norm(difference_array)
    norm = norm ** 2
    # Multiplying by gamma
    norm = norm * (-1) * gamma
    return np.exp(norm)

def polynomial_kernel(xj, xk, c = 2, d = 2):
    """
    Kernel Function, polynomial kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param c: mean of the polynomial kernel (np array)
    :param d: exponent of the polynomial (np array)
    :return: float64
    """
    #TODO DONE
    xj_transpose = np.transpose(xj)
    g = np.matmul(xj_transpose, xk)
    return (g + c) ** d

class SVM(object):

    def __init__(self, kernel_func=linear_kernel, lambda_param=.1):
        self.kernel_func = kernel_func
        self.lambda_param = lambda_param

    def train(self, inputs, labels):
        """
        train the model with the input data (inputs and labels),
        find the coefficients and constaints for the quadratic program and
        calculate the alphas

        :param inputs: inputs of data, a numpy array
        :param labels: labels of data, a numpy array
        :return: None
        """
        self.train_inputs = inputs
        self.train_labels = labels

        # constructing QP variables
        G = self._get_gram_matrix()
        Q, c = self._objective_function(G)
        A, b = self._inequality_constraint(G)

        # TODO: Uncomment the next line when you have implemented _get_gram_matrix(),
        # _inequality_constraints() and _objective_function().

        self.alpha = solve_QP(Q, c, A, b)[:self.train_inputs.shape[0]]

    def _get_gram_matrix(self):
        """
        Generate the Gram matrix for the training data stored in self.train_inputs.

        Recall that element i, j of the matrix is K(x_i, x_j), where K is the
        kernel function.

        :return: the Gram matrix for the training data, a numpy array
        """

        # TODO
        num_examples = self.train_labels.size
        gram_matrix = np.zeros((num_examples, num_examples))


        for i in range (num_examples):
            for j in range (num_examples):
                gram_matrix[i][j] = self.kernel_func(self.train_inputs[i], self.train_inputs[j])

        return gram_matrix

    def _objective_function(self, G):
        """
        Generate the coefficients on the variables in the objective function for the
        SVM quadratic program.

        Recall the objective function is:
        minimize (1/2)x^T Q x + c^T x

        :param G: the Gram matrix for the training data, a numpy array
        :return: two numpy arrays, Q and c which fully specify the objective function
        """

        # TODO

        ## Creating the "A" matrix
        num_examples = self.train_labels.size
        zero_matrix = np.zeros((num_examples, num_examples))
        gram_matrix = np.copy (G)
        top = np.hstack((gram_matrix, zero_matrix))
        bottom = np.hstack((zero_matrix, zero_matrix))

        Q = np.vstack ((top, bottom))

        ## Multiplying by 2 * lambda
        Q = Q * 2 * self.lambda_param


        ## Creating the "c" matrix
        c = np.zeros(num_examples * 2)

        for i in range (num_examples, 2 * num_examples):
            c[i] = 1/num_examples

        return Q, c

    def _inequality_constraint(self, G):
        """
        Generate the inequality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ax <= b.

        :param G: the Gram matrix for the training data, a numpy array
        :return: two numpy arrays, A and b which fully specify the constraints
        """

        # TODO (hint: you can think of x as the concatenation of all the alphas and
        # all the all the xi's; think about what this implies for what A should look like.).
        # DONE (but check)

        num_examples = self.train_labels.size

        ## Creating the "top half" of the A array, to take care of the ei>1-yi(G*alpha) constraint
        a_1 = np.zeros ((num_examples, num_examples))

        for i in range (num_examples):
            for j in range (num_examples):
                a_1[i][j] = self.train_labels[i] * G[i][j]

        a_2 = np.eye(num_examples)
        a_top = np.hstack((a_1, a_2))

        ## Negating the array
        a_top = -1 * a_top


        ## Creating the "bottom half" of the A array, to take care of the ei>0 constraint
        a_3 = np.zeros ((num_examples, num_examples))
        a_4 = np.eye (num_examples)
        a_bottom = np.hstack((a_3, a_4))

        ## Negating the array
        a_bottom = (-1) * a_bottom


        ## Concatenating into one array
        a = np.vstack((a_top,a_bottom))

        ## Creating the b-array
        b = np.zeros (2 * num_examples)
        for i in range (0, num_examples):
            b[i] = -1

        ## Returning the arrays
        return a, b

    def predict(self, input):
        """
        Generate predictions given input.

        :param input: 2D Numpy array. Each row is a vector for which we output a prediction.
        :return: A 1D numpy array of predictions.
        """

        #TODO
        num_training_examples = self.train_labels.size
        num_test_examples = input.shape[0]

        predictions = np.zeros(num_test_examples)

        for i in range (num_test_examples):
            sum = 0
            for j in range (num_training_examples):
                sum = sum + (self.alpha[j] * self.kernel_func(self.train_inputs[j], input[i]))

            # Setting the prediction to 1 if the sum is positive, -1 otherwise
            if sum >= 0:
                predictions[i] = 1
            else:
                predictions[i] = -1

        return predictions


    def accuracy(self, inputs, labels):
        """
        Calculate the accuracy of the classifer given inputs and their true labels.

        :param inputs: 2D Numpy array which we are testing calculating the accuracy of.
        :param labels: 1D Numpy array with the inputs corresponding true labels.
        :return: A float indicating the accuracy (between 0.0 and 1.0)
        """

        #TODO DONE
        predictions_array = self.predict(inputs)
        count_correct = 0

        for i in range (labels.size):
            if labels[i] == predictions_array[i]:
                count_correct = count_correct + 1

        return count_correct/labels.size
