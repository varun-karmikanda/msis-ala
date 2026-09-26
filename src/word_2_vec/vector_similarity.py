import os
from pathlib import Path
from gensim.models import KeyedVectors

def load_model(path: str):
    try:
        fast_model_path = os.path.expanduser(path)
        return KeyedVectors.load(fast_model_path, mmap='r')
    except Exception as e:
        print(f"Failed to load model in word2vec format: {e}")
    pass

def get_word_vector(model, word: str):
    word = word.lower()
    try:
        word_vector = model[word]
        assert len(word_vector) == 50
        return word_vector
    except KeyError:
        print(f"Word {word} not in model vocabulary.")

# TAGS
def preprocess_tags(path: str) -> Path:
    file_path = Path(os.path.expanduser(path))
    output_file_path = file_path.parent.parent / "processed" / file_path.name

    try:
        with open(file_path, "r") as file:
            file_content = file.read()

        comma_seperated_tags = file_content.replace("\n", ",").split(",")

        words = [tag.strip() for tag in comma_seperated_tags if tag.strip()]

        processed_content = ",".join(words)

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

        tags = file_content.split(",")
        return tuple(tags)

    except FileNotFoundError:
        print(f"File not found at path: {file_path}")


def build_tag_matrix(model, tags: tuple[str]):
    # tags_matrix = []    
    # for tag in tags:
    #     word_matrix = get_word_vector(model, tag)
    #     word_matrix_list = word_matrix.tolist()
    #     tags_matrix.append(word_matrix_list)

    tags_matrix = [get_word_vector(model, tag).tolist() for tag in tags]

    return tags, tuple(tags_matrix)

# TEXT
def preprocess_text(path: str) -> Path:
    pass

def get_text(path: str) -> tuple[str]:
    # preprocess_text(path)
    pass

def build_text_matrix(model, text: tuple[str]):
    # get_word_vector(word)
    pass

def similarity_matrix_vector_class(W, T):
    pass

def similarity_matrix_numpy(W, T):
    pass

if __name__ == "__main__":

    model_path = "assets/glove50/glove_50_fast.wordvectors"
    tags_file_path = "assets/word_2_vec/raw/tags.txt"
    text_file_path = ""

    model = load_model(model_path)

    tags = get_tags(tags_file_path)
    tag_name, T = build_tag_matrix(model, tags)

    text = get_text(text_file_path)
    text_matrix = build_text_matrix(model, text)

    sm_vector_class = similarity_matrix_vector_class(text_matrix, T)
    sm_numpy = similarity_matrix_numpy(text_matrix, T)
    