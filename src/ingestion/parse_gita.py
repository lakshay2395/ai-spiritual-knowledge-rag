import json
import os
import re


def parse_gita():
    # Paths relative to project root
    raw_dir = "data/raw/bhagavad-gita-as-it-is/en"
    processed_dir = "data/processed/bhagavad-gita-as-it-is"

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if not os.path.exists(raw_dir):
        print(
            f"Error: Could not find {raw_dir}. Make sure you are in the project root."
        )
        return

    # Iterate through numbered sub-folders (chapters)
    for folder_name in sorted(os.listdir(raw_dir)):
        # Skip non-numeric folders (like README.md, SUMMARY.md if they were there, and '0')
        if not folder_name.isdigit() or folder_name == "0":
            continue

        chapter_num = folder_name
        # Pad chapter number for filename: 1 -> 01.jsonl
        output_filename = f"{chapter_num.zfill(2)}.jsonl"
        output_path = os.path.join(processed_dir, output_filename)

        folder_path = os.path.join(raw_dir, folder_name)
        print(f"Processing Chapter {chapter_num}...")

        records = []

        # Sort files numerically, handling ranges like '16-18.md'
        def sort_key(x):
            if x == "README.md":
                return 999999
            name = x.split(".")[0]
            # Take the first number in the range or the number itself
            base_num = name.split("-")[0]
            return int(base_num) if base_num.isdigit() else 0

        # Iterate through verse files
        for filename in sorted(os.listdir(folder_path), key=sort_key):
            if not filename.endswith(".md") or filename == "README.md":
                continue

            verse_num = filename.replace(".md", "")
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract text under ### Translation:
            # It ends at the next ### heading or end of file
            match = re.search(
                r"### Translation:\s*(.*?)(?=\n### |\Z)", content, re.DOTALL
            )
            if match:
                translation_text = match.group(1).strip()
                # Remove bold marks if they wrap the content
                translation_text = translation_text.strip("*").strip()
                # Clean up whitespace
                translation_text = " ".join(translation_text.split()).strip()

                record = {
                    "text": translation_text,
                    "metadata": {"chapter": chapter_num, "verse": verse_num},
                }
                records.append(record)

        if records:
            with open(output_path, "w", encoding="utf-8") as out_f:
                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parse_gita()
