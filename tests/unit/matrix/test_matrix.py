from src.matrix.matrix import Matrix
import math

def test_init():
    try:
        Matrix()
        assert False, "Expected TypeError when the matrix is initialized with no parameters"
    except TypeError:
        pass

    try:
        Matrix(67)
        assert False, "Expected TypeError int cannot be iterated"
    except TypeError:
        pass

    try:
        Matrix([])
        assert False, "Expected TypeError when the matrix is initialized with empty list"
    except TypeError:
        pass

    try:
        Matrix([[1, 2, 3],[4, 5, "six"],["seven", 8, 9]])
        assert False, "Expected TypeError when the matrix is initialized with values other than int and float"
    except TypeError:
        pass

    try:
        Matrix([[67, 6767, 676767], [67, 6767]])
        assert False, "Expected TypeError when the matrix is initialized with values other than int and float"
    except TypeError:
        pass

def test_shape():
    m1 = Matrix([[67]])
    assert m1.shape() == (1, 1)

    m2 = Matrix([[1, 2, 3], [4, 5, 67], [8, 9, 0]])
    assert m2.shape() == (3, 3)

def test_length():
    m1 = Matrix([[1, 2, 3, 95, 51], [67, 1, 4, 5, 67], [8, 9, 0, 43, 78], [0, 0, 0, 0, 0], [67, 67, 67, 67, 67]])
    assert len(m1) == 25

    m2 = Matrix([[67]])
    assert len(m2) == 1

def test_neg():
    m1 = Matrix([[1, 45, -67], [35, 6, 0], [1, 53, 99]])
    m2 = -m1
    assert len(m2) == 9
    assert m2.matrix == ((-1, -45, 67), (-35, -6, 0), (-1, -53, -99))

def test_zeros():
    try:
        Matrix.zeros(-67)
        assert False, "Expected TypeError initializing a matrix with non positive dimention"
    except TypeError:
        pass

    try:
        Matrix.zeros(0)
        assert False, "Expected TypeError initializing a matrix with non positive dimention"
    except TypeError:
        pass

    m1 = Matrix.zeros(2, 3)
    assert m1.matrix == ((0, 0, 0), (0, 0, 0))
    assert all(element == 0 for row in m1.matrix for element in row)

    m2 = Matrix.zeros(67)
    assert m2.shape() == (67, 67)
    assert m2.matrix[67 - 1][67 - 1] == 0
    assert all(element == 0 for row in m2.matrix for element in row)


def test_ones():
    try:
        Matrix.ones(-67)
        assert False, "Expected TypeError initializing a matrix with non positive dimention"
    except TypeError:
        pass

    try:
        Matrix.ones(0)
        assert False, "Expected TypeError initializing a matrix with non positive dimention"
    except TypeError:
        pass

    m1 = Matrix.ones(4, 2)
    assert all(element == 1 for row in m1.matrix for element in row)

    m2 = Matrix.ones(67)
    assert m2.shape() == (67, 67)
    assert m2.matrix[67 - 1][67 - 1] == 1
    assert all(element == 1 for row in m2.matrix for element in row)

def test_uniform():
    try:
        Matrix.uniform(-67)
        assert False, "Expected TypeError initializing a matrix with non positive dimention"
    except TypeError:
        pass

    try:
        Matrix.uniform(0)
        assert False, "Expected TypeError initializing a matrix with non positive dimention"
    except TypeError:
        pass

    m1 = Matrix.uniform(4, 2)
    assert all(element >= 0 and element <= 5 for row in m1.matrix for element in row)

    m2 = Matrix.uniform(67)
    assert m2.shape() == (67, 67)
    assert all(element >= 0 and element <= 5 for row in m2.matrix for element in row)

def test_norm():
    m1 = Matrix.ones(67)
    assert m1.norm() == 67

    m2 = Matrix.ones(6, 7)
    assert m2.norm() == round(math.sqrt(6 * 7), 5)

    m3 = Matrix.zeros(67)
    assert m3.norm() == 0

    m4 = Matrix.uniform(67)
    assert m4.norm() > 0

    m5 = Matrix([[1, 2, 3], [4, 5, 67], [8, 9, 0]])
    assert m5.norm() == 68.47627

if __name__ == "__main__":
    test_init()
    test_shape()
    test_length()
    test_neg()
    test_zeros()
    test_ones()
    test_uniform()
    test_norm()