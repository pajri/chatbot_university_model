import json
import argparse

def extract_cells_from_ipynb(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook.get('cells', [])
    extracted_text = []

    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', 'unknown')
        source = ''.join(cell.get('source', []))
        extracted_text.append(f"#Cell {i} ({cell_type}):\n{source}")

    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(extracted_text))

    print(f"Extracted {len(cells)} cells to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract cells from a Jupyter Notebook (.ipynb) file.')
    parser.add_argument('input', help='Path to the input .ipynb file')
    parser.add_argument('output', help='Path to the output text file')

    args = parser.parse_args()
    extract_cells_from_ipynb(args.input, args.output)
