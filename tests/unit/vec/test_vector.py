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
    assert abs(v1.norm() - 4) < 0.00001

    v2 = Vec([6, 7, 67, 6767])
    assert v2.norm() - 6767.33796 < 0.00001

def test_mean():
    v1 = Vec([67.67, 6.7, 7.6, 670.67, -252.64])
    assert abs(v1.mean() - 100) < 0.00001

    v2 = Vec([10, -2, 8, 23, 6, 75])
    assert abs(v2.mean() - 20) < 0.00001

    v3 = Vec([67])
    assert abs(v3.mean() - 67) < 0.00001

    v4 = Vec([-67, -55, -90])
    assert abs(v4.mean() - (-212 / 3)) < 0.00001

    v5 = Vec([67, -67, 6, 7, -6, -7])
    assert v5.mean() == 0

    v6 = Vec([67])
    assert v6.mean() == v6.elements[0]

    v7 = Vec([67, 67, 67, 67, 67])
    assert v7.mean() == 67

    v8 = Vec([0, 0, 0])
    assert v8.mean() == 0

def test_demean():
    v1 = Vec([10, -2, 8, 23, 6, 75])
    v1_d = v1.demean()
    assert len(v1_d) == len(v1)
    # assert v1_d.elements == tuple([x - v1.mean() for x in v1.elements])
    assert v1_d.elements == (-10.0, -22.0, -12.0, 3.0, -14.0, 55.0)
    assert v1_d.mean() == 0
    assert sum(v1_d.elements) == 0

    v2 = Vec([67, 67, 67, 67, 67])
    v2_d = v2.demean()
    assert len(v2_d) == len(v2)
    assert v2_d.elements == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert v2_d.mean() == 0
    assert sum(v2_d.elements) == 0

def test_std():
    v1 = Vec([10, -2, 8, 23, 6, 75])
    v1_std = v1.std()
    assert v1_std - 25.68398 < 0.00001

    v2 = Vec([67, 67, 67, 67, 67])
    v2_std = v2.std()
    assert v2_std - 0 < 0.00001

    v3 = Vec([-67, -51, -95, -21])
    v3_std = v3.std()
    assert v3_std - 26.77219 < 0.00001

    v4 = Vec([0, 0, 0, 0])
    v4_std = v4.std()
    assert v4_std - 0 == 0


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
    test_mean()
    test_demean()
    test_std()