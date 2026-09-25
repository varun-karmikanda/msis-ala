import os
from pathlib import Path

def clean_tags_file(path: str) -> Path:
    try:
        file_path = Path(os.path.expanduser(path))
        print("Actual: ", file_path)
        print("Final: ", file_path)
        
        output_file_path = file_path.parent.parent / "processed" / file_path.name
        print("op: ", output_file_path)
        
        with open(file_path, "r") as file:
            content = file.read()
        
        print(content)
        print(type(content))
        
        cleaned_content = content.replace(" ", "")
        print(cleaned_content) 
        
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file_path, "w") as file:
            file.write(cleaned_content)
        
    except FileNotFoundError:
        print(f"File not found at path: {file_path}") 
    except Exception as e:
        print(f"Failed to load the file: {e}")
        
    return Path(output_file_path)

def get_all_tags(path: str) -> tuple[str]:
    file_path = clean_tags_file(path)
    
    with open(file_path, "r") as file:
        content = file.read()
        
    tags = content.split(",")
    
    return tuple(tags)

tags = get_all_tags("assets/word_2_vec/raw/tags.txt")
print(tags)
print(type(tags))


