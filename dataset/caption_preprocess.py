import re
from underthesea import text_normalize, word_tokenize

def preprocess_caption(text):
    text = text_normalize(text)
    text = word_tokenize(text, format="text")

    text = text.lower()

    # Keep letters/digits/underscore and whitespace only
    text = re.sub(r"[^\w\s_]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

input_filepath = "dataset/train/captions.txt"
output_filepath = "dataset/train/captions_preprocessed.txt"

with open(input_filepath, 'r', encoding='utf-8') as in_file, \
    open(output_filepath, 'w', encoding='utf-8') as out_file:

    for line in in_file:
        img_path, caption = line.strip().split('\t', 1)
        
        preprocessed_caption = preprocess_caption(caption)
        
        out_file.write(f"{img_path.lstrip('/')}\t{preprocessed_caption}\n")
        # Also remove the first slash from the image path to make it relative to the project root

print("Completed. Preprocessed captions saved to:", output_filepath)