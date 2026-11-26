import numpy as np
import numbers

class VecNumpy:
    def __init__(self, src):
        if len(src) == 0:
            raise TypeError("Vector cannot be empty!")
        
        for x in src:
            if not isinstance(x, numbers.Real):
                raise TypeError(f"Each scalar must be of type int or float: {type(x)}")
        
        elements = np.array(src, dtype=float)
        
        self.elements = elements
        
    def __repr__(self):
        return repr(self.elements)
    
    def __str__(self):
        return str(self.elements)
    
    def __len__(self):
        return len(self.elements)
    
    def __neg__(self):
        # return np.negative(self.elements)
        
        return VecNumpy(-self.elements)
    
    def __add__(self, other):
        if not isinstance(other, VecNumpy):
            raise TypeError(f"Expected a numpy vector: {type(other)}")
        
        if(len(self) != len(other)):
            raise TypeError("The vectors must be of the same length")
        
        return VecNumpy(self.elements + other.elements)
    
    def __sub__(self, other):
        if not isinstance(other, VecNumpy):
            raise TypeError(f"Expected a numpy vector: {type(other)}")
        
        if(len(self) != len(other)):
            raise TypeError("The vectors must be of the same length")
        
        return VecNumpy(self.elements - other.elements)
    
    def __mul__(self, scalar):
        if not isinstance(scalar, (int , float)):
            raise TypeError(f"Scalar must be int or float {type(scalar)}")
        
        return VecNumpy(self.elements * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __imul__(self, scalar):
        if not isinstance(scalar, (int , float)):
            raise TypeError(f"Scalar must be int or float {type(scalar)}")
        
        self.elements *= scalar
        return self
    
    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)
    
    def __iadd__(self, other):
        if not isinstance(other, VecNumpy):
            raise TypeError(f"Expected a numpy vector: {type(other)}")
        
        if(len(self) != len(other)):
            raise TypeError("The vectors must be of the same length")
        
        self.elements += other.elements
        return self
    
    @staticmethod
    def zeros(n):
        return VecNumpy(np.zeros(n))
    
    @staticmethod
    def ones(n):
        return VecNumpy(np.ones(n))
    
    @staticmethod
    def uniform(n):
        # return VecNumpy(np.random.default_rng().uniform(0, 100, 5))
        
        return VecNumpy(np.random.uniform(0, 100, n))
    
    def add(self, other):
        return self.__add__(other)
    
    def scale(self, scalar):
        return self.__mul__(scalar)
    
    def dot(self, other):
        return np.dot(self.elements, other.elements)
    
    def norm(self):
        return np.linalg.norm(self.elements)
    
    def cosine_similarity(self, other):
        denom = self.norm() * other.norm()
        if denom == 0:
            raise ValueError("Cosine similarity is undefined for a zero vector")

        return self.dot(other) / denom
    
    def mean(self):
        return np.mean(self.elements)
    
    def demean(self):
        return VecNumpy(self.elements - self.mean())
    
    def std(self):
        return np.std(self.elements)
    
    
v = VecNumpy((67, 67))
print(v)
print(str(v))
print(repr(v))
print(v.elements)
print(type(v.elements))
print(type(v))
print(len(v.elements))
print(len(v))
print(-v.elements)
print(v)

print(v + VecNumpy((1, 2)))
print(v + v)

sub = v - v
print(type(sub))
# print("v" * v)
print(v * 2)
print(2 * v)
v *=5
print(type(v))

print(VecNumpy.zeros(6))
print(VecNumpy.ones(7))
print(VecNumpy.uniform(5))

v2 = VecNumpy((1, 2))
v3 = VecNumpy((3, 4))
print(v2.add(v3))
print(v2.scale(3))

print(v2.dot(v3))
print(v2.norm())
print(v2.cosine_similarity(v3))
print(v2.mean())
print(v2.demean())
print(v2.std())

v1 = VecNumpy((1, 2))
v2 = VecNumpy((3, 4))
print(sum([v1, v2]))