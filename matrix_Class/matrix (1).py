import math
from math import sqrt
import numbers


def zeroes(height, width):
    """
    Creates a matrix of zeroes.
    """
    g = [[0.0 for _ in range(width)] for __ in range(height)]
    return Matrix(g)


def identity(n):
    """
    Creates a n x n identity matrix.
    """
    I = zeroes(n, n)
    for i in range(n):
        I.g[i][i] = 1.0
    return I


class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)
        self.w = len(grid[0])

    #
    # Primary matrix math methods
    #############################

    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        if not self.is_square():
            raise (ValueError, "Cannot calculate determinant of non-square matrix.")
        if self.h > 2:
            raise (ValueError, "Calculating determinant not implemented for matrices largerer than 2x2.")

        # TODO - your code here
        if self.h > 1:
            det = (self.g[0][0] * self.g[1][1]) - (self.g[1][0] * self.g[0][1])
            return det
        elif self.h == 0:
            return self.g[0][0]

    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        trace_result = 0
        if not self.is_square():
            raise (ValueError, "Cannot calculate the trace of a non-square matrix.")
        # TODO - your code here
        if self.h > 2:
            for i in range(len(self.g)):
                trace_result = trace_result +  self.g[i][i]
        elif self.h > 1:
            trace_result = self.g[0][0] + self.g[1][1]
        else:
            trace_result = self.g[0][0]
        return trace_result

    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        inverse = []
        inverse_one_element = []
        if not self.is_square():
            raise (ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise (NotImplementedError, "inversion not implemented for matrices larger than 2x2.")
        # TODO - your code here
        if self.h > 1:
            M_grid = self.g
            det = Matrix.determinant(Matrix(M_grid))
            inverse = [[(self.g[1][1]/det) , (-1 *self.g[0][1]/det) ], [(-1*self.g[1][0]/det)  , (self.g[0][0]/det)]]
            return Matrix(inverse)
        elif self.h == 1:
            inverse_one_element = [[1/self.g[0][0]]]
            return Matrix(inverse_one_element)
    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """
        result =[[0 for x in range(len(self.g))] for y in range(len(self.g[0]))]
        # TODO - your code here
        if not self.is_square():
            raise (ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise (NotImplementedError, "inversion not implemented for matrices larger than 2x2.")
        if self.h > 1:
            for i in range(len(self.g)):
                for j in range(len(self.g[0])):
                    result[i][j] = self.g[j][i]
        return Matrix(result)

    def is_square(self):
        return self.h == self.w

    #
    # Begin Operator Overloading
    ############################
    def __getitem__(self, idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]

    def __repr__(self):
        """
        Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    def __add__(self, other):
        """
        Defines the behavior of the + operator

        """
        Sum_Mat = [[0 for x in range(len(self.g))] for y in range(len(self.g[0]))]
        if self.h != other.h or self.w != other.w:
            raise (ValueError, "Matrices can only be added if the dimensions are the same")
            #
        # TODO - your code here
        #
        if self.h > 2:
            raise (NotImplementedError, "Addition not implemented for matrices larger than 2x2.")
        if self.h > 1:
            for i in range(len(self.g)):
                for j in range(len(self.g[0])):
                    Sum_Mat[i][j] = self.g[i][j] + other.g[i][j]
        return Matrix(Sum_Mat)

    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction)

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        #
        # TODO - your code here
        #

        neg_mat = [[0 for x in range(len(self.g))] for y in range(len(self.g[0]))]
        for i in range(len(self.g)):
            for j in range(len(self.g[0])):
                neg_mat[i][j] = -1 * self.g[i][j]
        return Matrix(neg_mat)

    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        #
        # TODO - your code here
        #
        sub_mat = [[0 for x in range(len(self.g))] for y in range(len(self.g[0]))]
        for i in range(len(self.g)):
            for j in range(len(self.g[0])):
                sub_mat[i][j] = self.g[i][j] - other.g[i][j]
        return Matrix(sub_mat)

    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        #
        # TODO - your code here
        #
        if self.w != other.h:
            raise (ValueError, "Dimensions are not compatible for multiplication")
        if self.h > 2:
            pass
        mul_mat = [[0 for x in range(len(self.g))] for y in range(len(other.g[0]))]
        for i in range(len(self.g)):
            for j in range(len(other.g[0])):
                for k in range(len(other.g)):
                    mul_mat[i][j] += self.g[i][k] * other.g[k][j]
        return Matrix(mul_mat)

    def __rmul__(self, other):
        """
        Called when the thing on the left of the * is not a matrix.

        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """
        result = [[0 for x in range(len(self.g))] for y in range(len(self.g[0]))]
        if isinstance(other, int):
            #
            # TODO - your code here
            #
            for i in range(len(self.g)):
                for j in range(len(self.g[0])):
                    result[i][j] = other * self.g[i][j]
        return Matrix(result)

          