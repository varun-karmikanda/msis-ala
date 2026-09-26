import os
import re
import string
from src.vec.vec import Vec
from pathlib import Path
from gensim.models import KeyedVectors

def load_model(path: str) -> KeyedVectors | None:
    try:
        fast_model_path = os.path.expanduser(path)
        return KeyedVectors.load(fast_model_path, mmap='r')
    except Exception as e:
        raise RuntimeError(f"Failed to load model in word2vec format: {e}")
    pass

def get_word_vector(model, word: str) -> list[float]:
    word = word.lower()
    try:
        word_vector = model[word]
        assert len(word_vector) == 50
        return word_vector
    except KeyError:
        raise ValueError(f"Word {word} not in model vocabulary.")

def get_stopwords(path: str) -> set[str]:
    try:
        with open(stopwords_file_path, "r") as file:
            return { line.strip() for line in file if line.strip() }
    except FileNotFoundError:
        print(f"File not found at path: {path}")

# TAGS
def preprocess_tags(path: str) -> Path:
    file_path = Path(os.path.expanduser(path))
    output_file_path = file_path.parent.parent / "processed" / file_path.name

    try:
        with open(file_path, "r") as file:
            file_content = file.read()

        comma_seperated_tags = file_content.replace("\n", ",").split(",")

        words = [tag.strip() for tag in comma_seperated_tags if tag.strip()]

        processed_content = "\n".join(words)

        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file_path, "w") as file:
            file.write(processed_content)

        return Path(output_file_path)

    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
    except Exception as e:
        print(f"Failed to load the file: {e}")


def get_tags(path: str) -> tuple[str]:
    file_path = preprocess_tags(path)

    try:
        with open(file_path, "r") as file:
            file_content = file.read()

        tags = file_content.split("\n")
        return tuple(tags)

    except FileNotFoundError:
        print(f"File not found at path: {file_path}")


def build_tag_matrix(model, tags: tuple[str]):
    tags_matrix = []    
    for tag in tags:
        try:
            word_matrix = get_word_vector(model, tag)
            word_matrix_list = word_matrix.tolist()
            tags_matrix.append(word_matrix_list)
        except ValueError:
            print(f"Word '{tag}' not in model vocabulary`")

    # tags_matrix = [get_word_vector(model, tag).tolist() for tag in tags]

    return tags, tuple(tags_matrix)

# TEXT
def preprocess_text(path: str, STOPWORDS: set[str]) -> Path:
    file_path = Path(os.path.expanduser(path))
    
    output_file_path = file_path.parent.parent / "processed" / file_path.name
    
    try:
        with open(file_path, "r") as file:
            file_content = file.read()
        
        print(f"Words before preprocessing {len(file_content.split(" "))}")
        
        # 1 - convert to lower case
        file_content = file_content.lower()
        
        # 2.1 - remove punctuation
        punctuation_chars = string.punctuation + '“”‘’—–…'
        punctuation_spaces = ' ' * len(punctuation_chars)
        file_content = file_content.translate(str.maketrans(punctuation_chars, punctuation_spaces))
        
        # 2.2 - remove the word 'hashtag' that is due to the hashtags of the post
        file_content = file_content.replace("hashtag", '')
        
        # 2.3 - remove all the numbers
        file_content = re.sub(r'\d+', '', file_content)
        
        # 2.4 - remove spaces
        file_content = re.sub(r'\s+', ' ', file_content)
        
        tokens = file_content.split(' ')
    
        # 3 - remove the stopwords from the tokens
        tokens = [token for token in tokens if token not in STOPWORDS]
        print(f"Number of tokens after removing stopwords: {len(tokens)}")
        
        # 4 - remove the tokens length less than equal to 2
        tokens = [token for token in tokens if(len(token)) > 2]
        print(f"Number of tokens after removing tokens length <= 2 : {len(tokens)}")
        
        with open(output_file_path, "w") as file:
            file.write("\n".join(tokens))
        
        return Path(output_file_path)
    
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")

def get_text(path: str, STOPWORDS: set[str]) -> tuple[str]:
    file_path = preprocess_text(path, STOPWORDS)
    
    try:
        with open(file_path, "r") as file:
            file_content = file.read()
        
        tokens = file_content.split("\n")
        
        return tuple(tokens)
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")

def build_text_matrix(model, tokens: tuple[str]):
    tokens_matrix = []
    in_vocob_tokens = []
    out_vocob_tokens = []
    
    for token in tokens:
        try:
            token_vector = get_word_vector(model, token)
            token_vector_list = token_vector.tolist()
            tokens_matrix.append(token_vector_list)
            in_vocob_tokens.append(token)
        except (ValueError, TypeError) as e:
            out_vocob_tokens.append(token)
            print(f"Skipping '{token}': {e}")
    
    return in_vocob_tokens, out_vocob_tokens, tuple(tokens_matrix)
            

def similarity_matrix_vector_class(W, T):
    pass

def similarity_matrix_numpy(W, T):
    pass

if __name__ == "__main__":

    model_path = "assets/glove50/glove_50_fast.wordvectors"
    tags_file_path = "assets/word_2_vec/raw/tags.txt"
    text_file_path = "assets/word_2_vec/raw/input.txt"
    stopwords_file_path = "assets/word_2_vec/raw/stop_words.txt"
    
    STOPWORDS = get_stopwords(stopwords_file_path)

    model = load_model(model_path)

    tags = get_tags(tags_file_path)
    tag_name, T = build_tag_matrix(model, tags)

    tokens = get_text(text_file_path, STOPWORDS)
    # tokens_matrix = build_text_matrix(model, tokens)
    in_vocob_tokens, out_vocob_tokens, W = build_text_matrix(model, tokens)
    print(f"Number of in-vocabulary tokens: {len(in_vocob_tokens)}")
    print(f"Number of out-of-vocabulary tokens: {len(out_vocob_tokens)}")
    print(out_vocob_tokens)
    
    print(f"Shape of T: ({len(T)} x {len(T[0])})")
    print(f"Shape of W: ({len(W)} x {len(W[0])})")
    

    # sm_vector_class = similarity_matrix_vector_class(W, T)
    # sm_numpy = similarity_matrix_numpy(W, T)
    
    
    
    