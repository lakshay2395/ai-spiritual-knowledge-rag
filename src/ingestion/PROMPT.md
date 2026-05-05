Lets write new separate parser separately for both folders (data/raw/bhagavad-gita-as-it-is and data/raw/mdbible)

For data/raw/mdbible:
- Refer to `by_book` folder contents
- Create individual JSONL files at book level like `01_Genesis.jsonl` with converting each numbered bullet points in file to individual JSONL rows. Verses are segregated by chapters as subheading like Chapter 1, Chapter 2, etc.
- The JSONL row structure should contain text and metadata (book name, chapter name, verse number (same as bulleted number))

For data/raw/bhagavad-gita-as-it-is:
- Refer to `en` folder contents. Consider all subfolders except `0`. Ignore README and SUMMARY file as well.
- In every numbered sub-folder, multiple numbered MD files exist. In file, refer only to text under `Translation` headings.
- Create individual JSONL files at sub-folder level like `01.jsonl`
- The JSONL row structure in each file should contain `Translation` as text and metadata (sub-folder number + chapter number)