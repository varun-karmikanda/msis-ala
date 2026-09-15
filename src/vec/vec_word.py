import os
from typing import Self
from src.vec.vec import Vec
from gensim.models import KeyedVectors

class Vec_Word(Vec):
    def __init__(self,  src=None) -> Self:
        if not src:
            raise TypeError("Vector cannot be empty!!")
        if isinstance(src, str):
            word = src
            src = self.get_word_vector(word).tolist()

            if src is None:
                raise ValueError(f"Word {word} not found in vocabulary.")

        super().__init__(src)

    @staticmethod
    def load_model():
        try:
            fast_model_path = os.path.expanduser("assets/glove50/glove_50_fast.wordvectors")
            return KeyedVectors.load(fast_model_path, mmap='r')
        except Exception as e:
            print(f"Failed to load model in word2vec format: {e}")
        return None

    @staticmethod
    def get_word_vector(word: str):
        model = Vec_Word.load_model()
        word = word.lower()
        try:
            v = model[word]
            assert len(v) == 50
            return v
        except KeyError:
            print(f"Word {word} not in model vocabulary.")

        return None

if __name__ == "__main__":
    v1 = Vec_Word("Varun")
    print(v1.elements)
    print(type(v1.elements))

    v2 = v1 * 5
    print(v2.elements)
