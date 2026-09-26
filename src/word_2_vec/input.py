import os
import re
import string
from pathlib import Path
from src.vec.vec import Vec

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

def process_text(path: str):
    file_path = Path(os.path.expanduser(path))
    print(file_path)
    
    output_file_path_matrix = file_path.parent.parent / "processed" / "matrix.txt"
    output_file_path_tokens_valid = file_path.parent.parent / "processed" / "tokens_valid.txt"
    output_file_path_tokens_skipped = file_path.parent.parent / "processed" / "tokens_skipped.txt"
    
    with open(file_path, "r") as file:
        file_content = file.read()
        
    # 1 - convert to lower case
    file_content = file_content.lower()
    
    # 2.1 - remove punctuation
    punctuation_chars = string.punctuation + '“”‘’—–…'
    punctuation_spaces = ' ' * len(punctuation_chars)
    file_content = file_content.translate(str.maketrans(punctuation_chars, punctuation_spaces))
    
    # 2.2 - remove the word 'hashtag' that is due to the hashtags of the post
    file_content = file_content.replace("hashtag", "")
    
    # 2.3 - remove spaces
    file_content = re.sub(r'\s+', ' ', file_content)
    
    # 2.4 - remove all the numbers
    file_content = re.sub(r'\d+', '', file_content)
    # print(file_content)
    
    tokens = file_content.split(' ')
    print(f"Number of tokens before preprocessing: {len(tokens)}")
    
    # no_stop = []
    # for token in tokens:
    #     if token not in STOPWORDS:
    #         no_stop.append(token)
    
    # 3 - remove the stopwords from the tokens
    token_without_stopwords = [token for token in tokens if token not in STOPWORDS]
    print(f"Number of tokens after removing stopwords: {len(token_without_stopwords)}")
    
    token_greater_than_2 = [token for token in token_without_stopwords if len(token) > 2]
    print(token_greater_than_2)
    print(f"Number of tokens after removing stopwords: {len(token_greater_than_2)}")
    
    matrix = []
    tokens_valid = []
    skipped_tokens = []
    
    for token in token_greater_than_2:
        try:
            matrix.append(Vec(token))
            tokens_valid.append(token)
        except (ValueError, TypeError) as e:
            skipped_tokens.append(token)
            print(f"Skipping '{token}': {e}")
    
    print(f"Number of tokens after removing stopwords: {len(token_greater_than_2) - len(skipped_tokens)}")
    
    # output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file_path_matrix, "w") as file:
        file.write("\n".join(" ".join(map(str, vec.elements)) for vec in matrix))
        
    with open(output_file_path_tokens_valid, "w") as file:
        file.write("\n".join(tokens_valid))
    
    with open(output_file_path_tokens_skipped, "w") as file:
        file.write("\n".join(skipped_tokens))

    
    
    
process_text("assets/word_2_vec/raw/input.txt")