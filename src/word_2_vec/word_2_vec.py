import os
from pathlib import Path

def process_tags_file(path: str) -> Path:
    try:
        file_path = Path(os.path.expanduser(path))
        
        output_file_path = file_path.parent.parent / "processed" / file_path.name
        
        with open(file_path, "r") as file:
            content = file.read()
            
        comma_seperated_tags = content.replace("\n", ",").split(",")
                
        words = [tags.strip() for tags in comma_seperated_tags if tags.strip()]
        
        cleaned_content = ",".join(words)
        
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file_path, "w") as file:
            file.write(cleaned_content)
        
    except FileNotFoundError:
        print(f"File not found at path: {file_path}") 
    except Exception as e:
        print(f"Failed to load the file: {e}")
        
    return Path(output_file_path)

def get_all_tags(path: str) -> tuple[str]:
    file_path = process_tags_file(path)
    
    with open(file_path, "r") as file:
        content = file.read()
        
    tags = content.split(",")
    
    return tuple(tags)

tags = get_all_tags("assets/word_2_vec/raw/tags.txt")
print(tags)


