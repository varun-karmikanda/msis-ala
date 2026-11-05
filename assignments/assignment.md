## Assignment 03 - GPT-2 Small Building Blocks

> Implement and explore the linear-algebra operations used by a small GPT-2-style
> transformer: token embeddings, layer normalization, scaled dot-product
> attention, and multi-head attention.

```text
msis-ala/
├── src/
│   └── gpt2_small/
│       └── algorithms.py
├── assignments/
│   └── assignment.md
├── README.md
└── requirements.txt
```

---
### Files

**GPT-2 small algorithms** ➜ [`algorithms.py`](../src/gpt2_small/algorithms.py)

---

**Clone the repository**
```
git clone git@github.com:varun-karmikanda/msis-ala.git
```

**Navigate to msis-ala**
```
cd msis-ala
```

### Make sure you have Python 3 or above and pip

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

**Run the GPT-2 small algorithms**
```
python3 -m src.gpt2_small.algorithms
```

[Back to Main README](../README.md)