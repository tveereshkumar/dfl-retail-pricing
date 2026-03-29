import os

def dump_src_to_file(src_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(src_dir):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != "__pycache__"]

            for file in files:
                file_path = os.path.join(root, file)

                # (Optional) only include .py files
                if not file.endswith(".py"):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    out.write(f"\n{'='*80}\n")
                    out.write(f"FILE: {file_path}\n")
                    out.write(f"{'='*80}\n\n")
                    out.write(content + "\n")

                except Exception as e:
                    out.write(f"\nERROR reading {file_path}: {e}\n")


if __name__ == "__main__":
    src_directory = "src"              # your source folder
    output_file = "all_files_dump.txt" # output file

    dump_src_to_file(src_directory, output_file)
    print(f"All files dumped into {output_file}")