import pytesseract
from pdf2image import convert_from_path
from service import get_summary_from_gpt
def extract_text_from_pdf(pdf_path):
    print(f"Starting the extraction process for: {pdf_path}")

    # Convert PDF to images
    print("Converting PDF pages to images...")
    pages = convert_from_path(pdf_path, dpi=300)
    print(f"Converted {len(pages)} pages to images.")

    # Initialize an empty string to hold the extracted text
    text_data = ''

    # Extract text from each page using Tesseract OCR
    for i, page in enumerate(pages):
        print(f"Extracting text from page {i + 1}...")
        text = pytesseract.image_to_string(page)
        text_data += text + '\n'  # Concatenate text from each page
        print(f"Extracted text from page {i + 1}.")

    print("Text extraction complete.")
    return text_data

# Specify the path to your PDF file
pdf_path = 'test.pdf'
extracted_text = extract_text_from_pdf(pdf_path)
# Print or save the extracted text
print("\nExtracted Text:")
print(extracted_text)

summary = get_summary_from_gpt(extracted_text)
# Optionally, save to a .txt file
with open('output.txt', 'w') as f:
    f.write(summary)
print("Extracted text has been saved to output.txt.")