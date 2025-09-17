# ImageToText 📖

A powerful command-line tool and Python library for extracting text from images using multiple OCR (Optical Character Recognition) engines. Similar to AudioToText but for visual content.

## Features 🚀

- **Multiple OCR Engines**: Support for Tesseract and EasyOCR
- **Batch Processing**: Process multiple images at once
- **Multiple Formats**: Support for PNG, JPG, TIFF, BMP, GIF, and PDF files
- **Language Support**: Multi-language OCR with 100+ supported languages
- **Output Formats**: Plain text, JSON, and CSV output options
- **Confidence Filtering**: Filter results by OCR confidence scores
- **Image Preprocessing**: Automatic image enhancement for better accuracy
- **PDF Support**: Extract text from multi-page PDF documents
- **Progress Tracking**: Real-time processing progress and statistics

## Installation 💾

### Prerequisites

First, install system dependencies:

**macOS (using Homebrew):**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download and install Tesseract from the [official releases](https://github.com/tesseract-ocr/tesseract/releases).

### Python Dependencies

Clone this repository and install Python dependencies:

```bash
git clone <repository-url>
cd ImageToText
pip install -r requirements.txt
```

Or install individual packages:
```bash
pip install pytesseract easyocr Pillow opencv-python PyMuPDF numpy
```

## Quick Start 🏁

### Basic Usage

Extract text from a single image:
```bash
python imagetotext.py image.png
```

Process multiple images:
```bash
python imagetotext.py *.jpg *.png
```

Save results to a file:
```bash
python imagetotext.py document.pdf -o extracted_text.txt
```

### Advanced Usage

Use EasyOCR engine:
```bash
python imagetotext.py image.jpg --engine easyocr
```

Process French text:
```bash
python imagetotext.py french_document.png --language fra
```

Output as JSON with confidence filtering:
```bash
python imagetotext.py *.png --format json --confidence-threshold 80 -o results.json
```

Batch process with detailed output:
```bash
python imagetotext.py examples/ --format csv -o batch_results.csv --verbose
```

## Command Line Options 📋

```
usage: imagetotext.py [-h] [-o OUTPUT] [--engine {tesseract,easyocr}]
                      [--language LANGUAGE] [--format {text,json,csv}]
                      [--confidence-threshold CONFIDENCE_THRESHOLD]
                      [--verbose] [--list-languages]
                      input_files [input_files ...]

positional arguments:
  input_files           Input image files or directories

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file (default: stdout)
  --engine {tesseract,easyocr}
                        OCR engine to use (default: tesseract)
  --language LANGUAGE   Language for OCR (default: eng)
  --format {text,json,csv}
                        Output format (default: text)
  --confidence-threshold CONFIDENCE_THRESHOLD
                        Minimum confidence threshold (0-100)
  --verbose, -v         Verbose output
  --list-languages      List available languages and exit
```

## Supported Languages 🌍

### Tesseract Languages
List available languages:
```bash
python imagetotext.py --list-languages --engine tesseract
```

Common language codes:
- `eng` - English
- `fra` - French
- `deu` - German
- `spa` - Spanish
- `ita` - Italian
- `por` - Portuguese
- `rus` - Russian
- `chi_sim` - Chinese Simplified
- `chi_tra` - Chinese Traditional
- `jpn` - Japanese
- `kor` - Korean

### EasyOCR Languages
EasyOCR supports 80+ languages. Common codes:
- `en` - English
- `fr` - French  
- `de` - German
- `es` - Spanish
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese

## Output Formats 📄

### Plain Text
Default format with metadata headers:
```
=== Image 1: document.png ===
Engine: tesseract, Language: eng
Confidence: 95.2%
Processing Time: 1.23s
Timestamp: 2024-01-15T10:30:00

This is the extracted text from the image...
```

### JSON
Structured data format:
```json
[
  {
    "text": "Extracted text content...",
    "confidence": 95.2,
    "engine": "tesseract",
    "language": "eng",
    "processing_time": 1.23,
    "image_path": "/path/to/image.png",
    "timestamp": "2024-01-15T10:30:00"
  }
]
```

### CSV
Tabular format for analysis:
```csv
image_path,engine,language,confidence,processing_time,timestamp,text
/path/to/image.png,tesseract,eng,95.2,1.23,2024-01-15T10:30:00,"Extracted text..."
```

## Examples 📚

### Processing a Screenshot
```bash
# Take a screenshot and save as screenshot.png
python imagetotext.py screenshot.png
```

### Extracting Text from a Receipt
```bash
python imagetotext.py receipt.jpg --engine easyocr --confidence-threshold 70
```

### Processing Multiple Documents
```bash
python imagetotext.py documents/*.pdf documents/*.png -o all_text.txt --verbose
```

### Multi-language Document
```bash
python imagetotext.py multilang_doc.png --language eng+fra --engine tesseract
```

### Batch Analysis
```bash
python imagetotext.py invoices/ --format csv --confidence-threshold 80 -o invoice_data.csv
```

## Performance Tips 🚀

### Image Quality
- Use high-resolution images (300+ DPI)
- Ensure good contrast between text and background
- Avoid blurry or rotated images when possible

### OCR Engine Selection
- **Tesseract**: Better for typed text, documents, and specific languages
- **EasyOCR**: Better for handwriting, mixed text, and natural scene text

### Processing Speed
- Tesseract is generally faster for simple documents
- EasyOCR may be slower but more accurate for complex images
- Use confidence thresholds to filter unreliable results

## Troubleshooting 🛠️

### Common Issues

**ImportError: No module named 'pytesseract'**
```bash
pip install pytesseract pillow
```

**TesseractNotFoundError**
- Ensure Tesseract is installed system-wide
- On macOS: `brew install tesseract`
- On Windows: Add Tesseract to PATH

**Poor OCR Accuracy**
- Try different OCR engines (`--engine easyocr`)
- Preprocess images (resize, enhance contrast)
- Use appropriate language settings
- Check image quality and resolution

**Memory Issues with Large Batches**
- Process images in smaller batches
- Use confidence thresholds to reduce output size
- Consider using lower resolution images

### Debug Mode
Enable verbose logging:
```bash
python imagetotext.py image.png --verbose
```

## Python API Usage 🐍

You can also use ImageToText as a Python library:

```python
from imagetotext import ImageToTextConverter, TesseractOCR, EasyOCREngine

# Basic usage
converter = ImageToTextConverter(engine='tesseract', language='eng')
result = converter.process_single_image('document.png')
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence}%")

# Batch processing
results = converter.process_batch(['img1.png', 'img2.jpg'])

# Use specific OCR engines
tesseract = TesseractOCR('fra')  # French
result = tesseract.extract_text('french_doc.png')

easyocr = EasyOCREngine(['en', 'es'])  # English and Spanish
result = easyocr.extract_text('mixed_lang.png')
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

### Development Setup
```bash
git clone <repository-url>
cd ImageToText
pip install -r requirements.txt
# Add development dependencies
pip install pytest black flake8
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black imagetotext.py
flake8 imagetotext.py
```

## License 📜

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Google's OCR engine
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Ready-to-use OCR with 80+ supported languages
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF processing library
- [Pillow](https://python-pillow.org/) - Python Imaging Library

## Changelog 📝

### v1.0.0
- Initial release with Tesseract and EasyOCR support
- Batch processing capabilities
- Multiple output formats (text, JSON, CSV)
- PDF support
- Multi-language OCR
- Image preprocessing features

---

**Happy text extraction! 📖✨**
