from src.vec.vec import Vec

# class TestVec:

def test_init():
    try:
        Vec()
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        Vec([])
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        Vec([5, 'six', "seven", 8])
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        Vec(67)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_is_Vec():
    v = Vec([1, 3, 5])
    assert isinstance(v, Vec)

def test_elements_is_tuple():
    v = Vec([1, 3, 5])
    assert isinstance(v.elements, tuple)

def test_repr():
    v = Vec([67, 67, 67])
    assert v.elements == (67, 67, 67)

def test_length():
    v1 = Vec([1, 3, 5, 7, 9])
    assert len(v1) == 5

    v2 = Vec([-1])
    assert len(v2) == 1

    v3 = Vec([0, 0, 0])
    assert len(v3) == 3

    try:
        Vec()
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_neg():
    v1 = Vec([6, 7])
    v2 = -v1
    assert v2.elements == (-6, -7)

    v3 = -v2
    assert v3.elements == (6, 7)

def test_add():
    v1 = Vec([23, -32])
    v2 = Vec([44, 99])
    v3 = v1 + v2
    assert v3.elements == (67, 67)

    try:
        v1 + (7, 9)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        v1 + Vec([67])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_sub():
    v1 = Vec([23, -32])
    v2 = Vec([44, 99])
    v3 = v1 - v2
    assert v3.elements == (-21, -131)

    try:
        v1 - (7, 9)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        v1 - Vec([67])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_mul():
    v1 = Vec([67, 63])
    v2 = v1 * 5
    assert v2.elements == (335, 315)

    try:
        v1 * 'v'
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_rmul():
    v1 = Vec([67, 63])
    v2 = 5 * v1
    assert v2.elements == (335, 315)

    try:
        'v' * v1
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_imul():
    v1 = Vec([67, 63])
    v1 *= 5
    assert v1.elements == (335, 315)

    try:
        v1 *= 'v'
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_radd():
    v1 = Vec([23, -32])
    v2 = Vec([44, 99])
    v3 = v2 + v1
    assert v3.elements == (67, 67)

    try:
        (7, 9) + v1 
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        Vec([67]) + v1 
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_iadd():
    v1 = Vec([23, -32])
    v2 = Vec([44, 99])
    v1 += v2
    assert v1.elements == (67, 67)

    try:
        v1 += (7, 9)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    try:
        v1 += Vec([67])
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_zeros():
    v1 = Vec.zeros(6)
    assert len(v1) == 6
    assert v1.elements == (0,0,0,0,0,0)

    try:
        Vec.zeros(-67)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_ones():
    v1 = Vec.ones(7)
    assert len(v1) == 7
    assert v1.elements == (1,1,1,1,1,1,1)

    try:
        Vec.ones(-67)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_uniform():
    v1 = Vec.uniform(3)
    assert len(v1) == 3
    assert all(element > 0 for element in v1.elements)

    try:
        Vec.uniform(-67)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_norm():
    v1 = Vec([-3, 2, -1, 1, -1])
    assert Vec.norm(v1) == 4

    v2 = Vec([6, 7, 67, 6767])
    assert Vec.norm(v2) == 6767.33796


if __name__ == "__main__":
    test_init()
    test_is_Vec()
    test_elements_is_tuple()
    test_length()
    test_repr()
    test_neg()
    test_add()
    test_sub()
    test_mul()
    test_rmul()
    test_imul()
    test_radd()
    test_iadd()
    test_zeros()
    test_ones()
    test_uniform()
    test_norm()