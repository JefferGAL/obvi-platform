import pypandoc
import os
import tempfile
from typing import Optional

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentConverter:
    """Handles document conversion using pypandoc"""
    
    def __init__(self):
        try:
            # Test if pypandoc is available
            pypandoc.get_pandoc_version()
            self.pandoc_available = True
        except:
            self.pandoc_available = False

    def _sanitize_markdown(self, markdown_content: str) -> str:
        """Sanitizes markdown to remove characters known to break Pandoc."""
        # This is a simple "rule." More complex rules acan be added as needed.
        # e.g. escaping special characters or handling complex URLs.
        return markdown_content.encode('ascii', 'ignore').decode('ascii')

    def convert_markdown_to_docx(self, markdown_content: str, output_filename: str = None) -> bytes:
        """Convert markdown content to DOCX format with lotsa error handling."""
        if not self.pandoc_available:
            raise RuntimeError("Document conversion not available - pypandoc not installed")
        
        try:
            # The markdown content is sanitized before conversion.
            sanitized_content = self._sanitize_markdown(markdown_content)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_md:
                temp_md.write(sanitized_content)
                temp_md_path = temp_md.name
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_docx:
                temp_docx_path = temp_docx.name
            
            pypandoc.convert_file(
                temp_md_path, 
                'docx', 
                outputfile=temp_docx_path,
                extra_args=['--standalone']
            )
            
            with open(temp_docx_path, 'rb') as f:
                docx_content = f.read()
            
            os.unlink(temp_md_path)
            os.unlink(temp_docx_path)
            
            return docx_content
            
        except Exception as e:
            # ANNOTATION: The error logging is now much more detailed. It will log the
            # specific error message from the Pandoc tool itself, which is crucial for debugging.
            logger.error(f"Pandoc DOCX conversion failed. Error: {str(e)}")
            # For more detailed debugging, you might log the full traceback
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"DOCX conversion failed. Pandoc error: {str(e)}")

    def convert_markdown_to_pdf(self, markdown_content: str, filename: str = None) -> bytes:
        """Convert markdown to PDF using pypandoc"""
        try:
            # Use basic pandoc options without unsupported margin flags
            extra_args = [
                '--pdf-engine=xelatex',
                '-V', 'geometry:margin=1in'
            ]
            
            pdf_content = pypandoc.convert_text(
                markdown_content,
                'pdf',
                format='md',
                extra_args=extra_args
            )
            
        except Exception as e:
            raise RuntimeError(f"PDF conversion failed: {str(e)}")