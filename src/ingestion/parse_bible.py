import os
import json
import re

def parse_bible():
    # Paths relative to project root
    raw_dir = 'data/raw/mdbible/by_book'
    processed_dir = 'data/processed/mdbible'
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Check if we are in the right directory
    if not os.path.exists(raw_dir):
        print(f"Error: Could not find {raw_dir}. Make sure you are in the project root.")
        return

    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith('.md'):
            continue
        
        # Example: 01_Genesis.md -> Genesis
        book_name = filename.replace('.md', '').split('_', 1)[-1]
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(processed_dir, filename.replace('.md', '.jsonl'))
        
        print(f"Processing {filename}...")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by chapters
        # Use a non-capturing group for the split to keep the chapter text but keep chapter num
        chapters = re.split(r'## Chapter (\d+)', content)
        # First element is usually the book title heading, ignore it
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for i in range(1, len(chapters), 2):
                chapter_num = chapters[i]
                chapter_text = chapters[i+1]
                
                # Find verses: look for lines starting with "Number. "
                # We use re.MULTILINE to match start of lines
                # We also want to capture until the next verse or end of text
                # Bible verses are usually single paragraphs in this format
                verses = re.findall(r'^(\d+)\. (.*?)(?=\n\d+\. |\n## |\Z)', chapter_text, re.DOTALL | re.MULTILINE)
                
                for verse_num, verse_text in verses:
                    # Clean up the verse text (remove extra whitespace/newlines)
                    verse_text = ' '.join(verse_text.split()).strip()
                    record = {
                        "text": verse_text,
                        "metadata": {
                            "book": book_name,
                            "chapter": chapter_num,
                            "verse": verse_num
                        }
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parse_bible()
