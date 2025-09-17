# Example Images for Testing

This directory should contain sample images for testing the ImageToText tool.

## Recommended Test Images

### Basic Text Recognition
- **screenshot.png**: Screenshot of text from a webpage or document
- **typed_document.jpg**: Photo of a typed document
- **handwritten_note.jpg**: Photo of handwritten text (challenging)

### Multi-language Testing
- **french_text.png**: French text sample
- **spanish_text.jpg**: Spanish text sample  
- **mixed_languages.png**: Document with multiple languages

### Different Formats
- **invoice.pdf**: PDF document with text and numbers
- **receipt.jpg**: Photo of a receipt
- **business_card.png**: Business card image

### Challenging Cases
- **low_quality.jpg**: Blurry or low-resolution image
- **angled_photo.jpg**: Photo taken at an angle
- **colorful_background.png**: Text on complex background

## Creating Test Images

You can create test images by:

1. Taking screenshots of text on your computer
2. Photographing documents with your phone
3. Downloading sample images from the internet
4. Using online text-to-image generators

## Example Usage

Once you have sample images in this directory, test them with:

```bash
# Basic usage
python ../imagetotext.py screenshot.png

# Batch process all images
python ../imagetotext.py *.png *.jpg -o results.txt

# Use different OCR engine
python ../imagetotext.py invoice.pdf --engine easyocr

# Process with confidence filtering
python ../imagetotext.py *.jpg --confidence-threshold 80 --format json
```

## Expected Results

The tool should be able to:
- Extract text from clear images with 90%+ confidence
- Handle multiple languages when specified
- Process PDF files page by page
- Output in various formats (text, JSON, CSV)
- Provide confidence scores and processing times
