from typing import Self
import random
import math

class Matrix:
    def __init__(self, src=None) -> Self:
        if not src:
            raise TypeError("Vector cannot be empty!")
        else:
            matrix = tuple(tuple(row) for row in src)

        for row in matrix:
            for element in row:
                if not isinstance(element, (int, float)):
                    raise TypeError(f"Must be of type int or float {type(element)}")

        if any(len(row) != len(matrix[0]) for row in matrix):
            raise TypeError("All row must have same length!")

        self.matrix = matrix

    def __repr__(self):
        return repr(self.matrix)

    def shape(self):
        return (len(self.matrix), len(self.matrix[0]))

    def __len__(self):
        return len(self.matrix)

    def __neg__(self):
        # negated_matrix = []
        # for row in self.matrix:
        #     neagted_row = []
        #     for element in row:
        #         neagted_row.append(-element)
        #     negated_matrix.append(neagted_row)
        # print(negated_matrix)
        negated_matrix = [[-element for element in row] for row in self.matrix]
        return Matrix(negated_matrix)
    
    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __imul__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __iadd__(self, other):
        pass

    @staticmethod
    def zeros(n: int) -> Self:
        # zeross = []
        # for i in range(n):
        #     zeross_row = []
        #     for i in range(n):
        #         zeross_row.append(0)
        #     zeross.append(zeross_row)

        # zeross = [[0 for i in range(n)] for i in range(n)]

        if n <= 0:
            raise TypeError("The dimention of the matrix must be positive")

        zeross = [[0] * n for i in range(n)]
        return Matrix(zeross)

    @staticmethod
    def zeros(rows: int, columns: int | None = None) -> Self:
        # zeroos = []
        # for row in range(rows):
        #     zeroos_row = []
        #     for element in range(columns):
        #         zeroos_row.append(0)
        #     zeroos.append(zeroos_row)

        if columns == None:
            columns = rows

        if rows <= 0 or columns <= 0:
            raise TypeError("The dimention of the matrix must be positive")

        # zeroos = [[0 for element in range(columns)] for row in range(rows)]

        return Matrix([[0] * columns for _ in range(rows)])


    @staticmethod
    def ones(rows: int, columns: int | None = None) -> Self:
        if columns == None:
            columns = rows

        if rows <= 0 or columns <= 0:
            raise TypeError("The dimention of the marix must be positive")

        return Matrix([[1] * columns for _ in range(rows)])

    # @staticmethod
    def uniform(rows: int, columns: int | None = None) -> Self:
        if columns == None:
            columns = rows
            
        if rows <= 0 or columns <= 0:
            raise TypeError("The dimention of the matrix must be positive")
        
        return Matrix([[round(random.uniform(1, 5), 5) for _ in range(columns)] for _ in range(rows)])

    @staticmethod
    def norm(self: Self) -> float:
        # sum_square = 0
        # for row in self.matrix:
        #     for element in row:
        #         sum_square += element ** 2

        sum_square = sum(element ** 2 for row in self.matrix for element in row)
        return round(math.sqrt((sum_square)), 5)

if __name__ == "__main__":
    m1 = Matrix([
            [67, 2, 3],
            [4, 5, 63],
            [7, 51, 9]
    ])

    # print(m1)
    # print(type(m1))
    # print(type(m1.matrix))

    # print(m1.shape())

    # v1 = time.time()
    # m2 = -m1
    # v2 = time.time()
    # print(f"Time for 2 for loops: {(v2 - v1) * 1000}")

    # v1 = time.time()
    # m3 = -m1
    # v2 = time.time()
    # print(f"Time list comprehension: {(v2 - v1) * 1000}")

    # print(Matrix.zeros(6))

    # print(Matrix.ones(4))

    # print(Matrix.uniform(5))

    # print("Euclud norm = ", Matrix.norm(m1))

    # print(Matrix.zeros(6, 7))

    # print(Matrix.ones(6, 7))

    # print(Matrix.uniform(2, 3))

    # vv1 = Matrix.uniform(6, 7)
    # print(vv1)
    # print(Matrix.norm(vv1))

    print(m1)
    print(Matrix.norm(m1))