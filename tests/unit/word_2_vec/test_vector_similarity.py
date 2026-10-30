import numpy as np

from src.word_2_vec.vector_similarity import (
    load_model,
    get_stopwords,
    get_tags,
    build_tag_matrix,
    get_text,
    build_text_matrix,
    similarity_matrix_vector_class,
    similarity_matrix_numpy
)

def test_tag_matrix_shape(T: tuple):
    assert np.array(T).shape == (20, 50)
    
def test_text_matrix_shape(W: tuple, in_vocob_tokens: list): 
    assert np.array(W).shape == (len(in_vocob_tokens), 50)
    
def test_similarity_matrix_shape(S: list, in_vocob_tokens: list):
    assert np.array(S).shape == (len(in_vocob_tokens), 20)
    
def test_similarity_matrix_numpy_shape(S_numpy: list, in_vocob_tokens: list):
    assert np.array(S_numpy).shape == (len(in_vocob_tokens), 20)
    

if __name__ == "__main__":
    
    model_path = "assets/glove50/glove_50_fast.wordvectors"
    tags_file_path = "assets/word_2_vec/raw/tags.txt"
    text_file_path = "assets/word_2_vec/raw/input.txt"
    stopwords_file_path = "assets/word_2_vec/raw/stop_words.txt"
    
    model = load_model(model_path)
    
    STOPWORDS = get_stopwords(stopwords_file_path)
    
    tags = get_tags(tags_file_path)
    tag_name, T = build_tag_matrix(model, tags)
    
    tokens = get_text(text_file_path, STOPWORDS)
    in_vocob_tokens, out_vocob_tokens, W = build_text_matrix(model, tokens)
    
    S = similarity_matrix_vector_class(W, T)
    S_numpy = similarity_matrix_numpy(W, T)
    
    test_tag_matrix_shape(T)
    test_text_matrix_shape(W, in_vocob_tokens)
    test_similarity_matrix_shape(S, in_vocob_tokens)
    test_similarity_matrix_numpy_shape(S_numpy, in_vocob_tokens)
    
    
    