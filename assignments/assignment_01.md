## Assignment 1

> **Please refer the below steps to run and test the Vec class**

```text
msis-ala/
├── assets/
│   └── glove50/
├── src/
│   └── vec/
│       └── vec.py
├── tests/
│   └── unit/
│       └── vec/
│           └── test_vector.py
├── .gitignore
├── README.md
└── requirements.txt
```

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

**To run src/vec/vec.py**
```
python3 -m src.vec.vec
```

**To run tests/unit/vec/test_vector.py**
```
python3 -m tests.unit.vec.test_vector
```

**To run the pytest version for tests/unit/vec/test_vector.py**
```
python3 -m pytest tests/unit/vec/test_vector.py -v
```

[Back to Main README](../README.md)
