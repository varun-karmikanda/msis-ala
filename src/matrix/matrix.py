from typing import Self

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

if __name__ == "__main__":
    m1 = Matrix([
        [67, 2, 3],
        [4, 5, 63],
        [7, 51, 9]
    ])

    print(m1)
    print(type(m1))
    print(type(m1.matrix))

    print(m1.shape())

