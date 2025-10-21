import os
import sys
from fpdf import FPDF

def convert_to_pdf(source_folder, destination_folder):
    """
    Converts all .py, .js, and .css files in a source folder to PDF files
    and saves them in a new destination folder.
    
    Args:
        source_folder (str): The path to the folder containing the files.
        destination_folder (str): The path to the folder where PDFs will be saved.
    """
    # Check if the source folder exists
    if not os.path.isdir(source_folder):
        print(f"Error: Source folder '{source_folder}' not found.")
        return

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    print(f"Destination folder created at: {os.path.abspath(destination_folder)}")

    # Check for relevant files in the source folder
    relevant_files = [f for f in os.listdir(source_folder) if f.endswith(('.py', '.js', '.css'))]
    if not relevant_files:
        print(f"No .py, .js, or .css files found in '{source_folder}'.")
        return

    # Iterate and convert each relevant file
    for filename in relevant_files:
        file_path = os.path.join(source_folder, filename)
        
        # Determine the new filename
        if filename.endswith('.py'):
            pdf_filename = filename.replace('.py', '-py.pdf')
        elif filename.endswith('.js'):
            pdf_filename = filename.replace('.js', '-js.pdf')
        elif filename.endswith('.css'):
            pdf_filename = filename.replace('.css', '-css.pdf')
        else:
            continue # Skip unknown file types

        pdf_file_path = os.path.join(destination_folder, pdf_filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as source_file:
                content = source_file.read()

            # Create a PDF document
            pdf = FPDF('P', 'mm', 'A4')
            pdf.add_page()
            pdf.set_font('Courier', '', 10)
            
            # Add content to the PDF, handling newlines
            for line in content.splitlines():
                pdf.cell(w=0, h=5, txt=line, ln=1)
                
            pdf.output(pdf_file_path)
            
            print(f"Successfully converted '{filename}' to '{pdf_filename}'.")
        
        except Exception as e:
            print(f"Error converting '{filename}': {e}")
    
    print("\nConversion process complete.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python converter.py <source_folder> <destination_folder>")
        print("Example: python converter.py . pr2md")
        sys.exit(1)

    source = sys.argv[1]
    destination = sys.argv[2]
    convert_to_pdf(source, destination)