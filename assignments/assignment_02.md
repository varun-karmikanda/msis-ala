## Assignment 02 — Word2Vec Semantic Tagging

> Follow the steps below to run and test semantic tagging with GloVe and NumPy.

```text
msis-ala/
├── assets/
│   ├── glove50/
│   └── word_2_vec
├── src/
│   └── word_2_vec/
│       └── vector_similarity.py
├── tests/
│   └── unit/
│       └── word_2_vec/
│           └── test_vector_similarity.py
├── .gitignore
├── README.md
└── requirements.txt
```
---
### Files

**Word2Vec semantic tagging implementation** ➜ [`vector_similarity.py`](../src/word_2_vec/vector_similarity.py)

**Word2Vec semantic tagging tests** ➜ [`test_vector_similarity.py`](../tests/unit/word_2_vec/test_vector_similarity.py)

---

**Clone the repository**
```
git clone git@github.com:varun-karmikanda/msis-ala.git
```

**Navigate to msis-ala**
```
cd msis-ala
```

### Make sure you have the python3 or above and pip

**Create a virtual environment**
```
python3.12 -m venv venv
```

**Activate the virtual environment**
```
source venv/bin/activate
```

**Install the dependencies**
```
pip install -r requirements.txt
```

**Run the Word2Vec semantic tagging implementation**
```
python3 -m src.word_2_vec.vector_similarity
```

**Run the Word2Vec semantic tagging tests**
```
python3 -m tests.unit.word_2_vec.test_vector_similarity
```

### References

- [Word2Vec and semantic tagging sources](../assets/word_2_vec/citation.md)
- [Stopwords list used for text preprocessing](https://gist.github.com/sebleier/554280)

[Back to Main README](../README.md)
