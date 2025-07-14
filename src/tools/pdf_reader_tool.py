import sys
import fitz
from pathlib import Path
from docling.document_converter import DocumentConverter

class PDFReaderTool:
    """Tool for reading and extracting structured content from PDF files using Docling."""
    

    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"File '{self.pdf_path}' does not exist")
        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError("File must be a PDF")

    def extract_text_blocks(self):
        """Extracts all text blocks from the PDF."""
        text_blocks = []
        doc = fitz.open(str(self.pdf_path))
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()  # block[4] is the text
                if text:
                    text_blocks.append(f"--- Page {page_num} ---\n{text}")
        return text_blocks

    def extract_images(self, output_dir="extracted_images"):
        """Extracts all images from the PDF and saves them to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        images_info = []
        doc = fitz.open(str(self.pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = output_dir / f"page{page_num+1}_img{img_index+1}.{image_ext}"
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                images_info.append(str(image_filename))
        return images_info

    def extract_all(self):
        text = self.extract_text_blocks()
        images = self.extract_images()
        tables = []  # Table extraction not natively supported by PyMuPDF
        metadata = self.get_metadata()
        return {
            "text": text,
            "tables": tables,
            "images": images,
            "metadata": metadata
        }

    def get_metadata(self):
        doc = fitz.open(str(self.pdf_path))
        return doc.metadata

    def print_structured_content(self, content):
        print("=" * 80)
        print("METADATA")
        print("=" * 80)
        for key, value in content['metadata'].items():
            print(f"{key}: {value}")

        print("\n" + "=" * 80)
        print("TEXT CONTENT")
        print("=" * 80)
        for idx, text_block in enumerate(content['text'], 1):
            print(f"\n--- Text Block {idx} ---\n")
            print(text_block)

        print("\n" + "=" * 80)
        print("TABLES")
        print("=" * 80)
        if content['tables']:
            for idx, table in enumerate(content['tables'], 1):
                print(f"\n--- Table {idx} ---\n")
                print(table)
        else:
            print("No tables extracted (table extraction not supported by PyMuPDF).")

        print("\n" + "=" * 80)
        print("IMAGES")
        print("=" * 80)
        if content['images']:
            for idx, image_path in enumerate(content['images'], 1):
                print(f"Image {idx}: {image_path}")
        else:
            print("No images extracted.")

if __name__ == "__main__":
    pdf_path = r"C:\Users\bekal\OneDrive\Desktop\AI4SE\Litterature\RAG\Chain-of-RAG.pdf"
    try:
        reader = PDFReaderTool(pdf_path)
        print("Extracting content from PDF...")
        content = reader.extract_all()
        reader.print_structured_content(content)
        print("\nPDF PROCESSING COMPLETED SUCCESSFULLY")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)