#!/usr/bin/env python3
"""
ImageToText: Extract text from images using various OCR engines
A versatile tool for optical character recognition with multiple engine support
"""

import argparse
import logging
import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time
from dataclasses import dataclass, asdict
from datetime import datetime

# Import OCR libraries with graceful fallbacks
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    Image = None
    ImageEnhance = None
    print("Warning: Tesseract OCR not available. Install with: pip install pytesseract pillow")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Install with: pip install easyocr")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PDF support not available. Install with: pip install PyMuPDF")


@dataclass
class OCRResult:
    """Container for OCR results"""
    text: str
    confidence: float
    engine: str
    language: str
    processing_time: float
    image_path: str
    timestamp: str


class ImagePreprocessor:
    """Image preprocessing utilities for better OCR accuracy"""
    
    @staticmethod
    def enhance_image(image, enhance_contrast: bool = True, 
                     enhance_sharpness: bool = True):
        """Apply image enhancements to improve OCR accuracy"""
        if not TESSERACT_AVAILABLE:
            return image
            
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
        
        if enhance_sharpness:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
        
        return image
    
    @staticmethod
    def convert_to_grayscale(image):
        """Convert image to grayscale"""
        if not TESSERACT_AVAILABLE:
            return image
        return image.convert('L')
    
    @staticmethod
    def apply_threshold(image, threshold: int = 128):
        """Apply binary threshold to image"""
        if not TESSERACT_AVAILABLE:
            return image
        return image.point(lambda x: 0 if x < threshold else 255, '1')


class TesseractOCR:
    """Tesseract OCR engine wrapper"""
    
    def __init__(self, language: str = 'eng'):
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract OCR not available")
        self.language = language
    
    def extract_text(self, image_path: str, config: str = '') -> OCRResult:
        """Extract text using Tesseract OCR"""
        start_time = time.time()
        
        try:
            image = Image.open(image_path)
            
            # Preprocess image
            preprocessor = ImagePreprocessor()
            image = preprocessor.enhance_image(image)
            image = preprocessor.convert_to_grayscale(image)
            
            # Extract text with confidence
            data = pytesseract.image_to_data(image, lang=self.language, 
                                           config=config, output_type=pytesseract.Output.DICT)
            
            # Filter out low confidence detections
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=self.language, config=config)
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence,
                engine='tesseract',
                language=self.language,
                processing_time=processing_time,
                image_path=image_path,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logging.error(f"Tesseract OCR failed for {image_path}: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine='tesseract',
                language=self.language,
                processing_time=time.time() - start_time,
                image_path=image_path,
                timestamp=datetime.now().isoformat()
            )


class EasyOCREngine:
    """EasyOCR engine wrapper"""
    
    def __init__(self, languages: List[str] = ['en']):
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available")
        self.languages = languages
        self.reader = easyocr.Reader(languages)
    
    def extract_text(self, image_path: str) -> OCRResult:
        """Extract text using EasyOCR"""
        start_time = time.time()
        
        try:
            results = self.reader.readtext(image_path, detail=1)
            
            # Extract text and average confidence
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                text_parts.append(text)
                confidences.append(confidence)
            
            combined_text = '\n'.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=combined_text.strip(),
                confidence=avg_confidence * 100,  # Convert to percentage
                engine='easyocr',
                language=','.join(self.languages),
                processing_time=processing_time,
                image_path=image_path,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logging.error(f"EasyOCR failed for {image_path}: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine='easyocr',
                language=','.join(self.languages),
                processing_time=time.time() - start_time,
                image_path=image_path,
                timestamp=datetime.now().isoformat()
            )


class PDFTextExtractor:
    """Extract text from PDF files"""
    
    @staticmethod
    def extract_from_pdf(pdf_path: str, ocr_engine: str = 'tesseract', 
                        language: str = 'eng') -> List[OCRResult]:
        """Extract text from PDF using OCR on each page"""
        if not PDF_SUPPORT:
            raise ImportError("PDF support not available")
        
        results = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Convert page to image
            mat = fitz.Matrix(2, 2)  # Zoom factor for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Save temporary image
            temp_image_path = f"/tmp/pdf_page_{page_num}.png"
            with open(temp_image_path, "wb") as f:
                f.write(img_data)
            
            # Extract text using selected OCR engine
            if ocr_engine == 'tesseract' and TESSERACT_AVAILABLE:
                ocr = TesseractOCR(language)
                result = ocr.extract_text(temp_image_path)
            elif ocr_engine == 'easyocr' and EASYOCR_AVAILABLE:
                ocr = EasyOCREngine([language])
                result = ocr.extract_text(temp_image_path)
            else:
                continue
            
            result.image_path = f"{pdf_path} (page {page_num + 1})"
            results.append(result)
            
            # Clean up temporary file
            os.remove(temp_image_path)
        
        doc.close()
        return results


class ImageToTextConverter:
    """Main converter class"""
    
    def __init__(self, engine: str = 'tesseract', language: str = 'eng'):
        self.engine = engine
        self.language = language
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.pdf']
    
    def process_single_image(self, image_path: str) -> OCRResult:
        """Process a single image file"""
        file_ext = Path(image_path).suffix.lower()
        
        if file_ext == '.pdf':
            if PDF_SUPPORT:
                results = PDFTextExtractor.extract_from_pdf(
                    image_path, self.engine, self.language
                )
                # Combine all pages into one result
                if results:
                    combined_text = '\n\n--- Page Break ---\n\n'.join(
                        [r.text for r in results if r.text]
                    )
                    avg_confidence = sum(r.confidence for r in results) / len(results)
                    total_time = sum(r.processing_time for r in results)
                    
                    return OCRResult(
                        text=combined_text,
                        confidence=avg_confidence,
                        engine=self.engine,
                        language=self.language,
                        processing_time=total_time,
                        image_path=image_path,
                        timestamp=datetime.now().isoformat()
                    )
            else:
                raise ImportError("PDF support not available")
        
        # Handle regular image files
        if self.engine == 'tesseract' and TESSERACT_AVAILABLE:
            ocr = TesseractOCR(self.language)
            return ocr.extract_text(image_path)
        elif self.engine == 'easyocr' and EASYOCR_AVAILABLE:
            ocr = EasyOCREngine([self.language])
            return ocr.extract_text(image_path)
        else:
            raise ValueError(f"OCR engine '{self.engine}' not available")
    
    def process_batch(self, input_paths: List[str]) -> List[OCRResult]:
        """Process multiple image files"""
        results = []
        
        for path in input_paths:
            if not os.path.exists(path):
                logging.warning(f"File not found: {path}")
                continue
            
            file_ext = Path(path).suffix.lower()
            if file_ext not in self.supported_formats:
                logging.warning(f"Unsupported format: {path}")
                continue
            
            try:
                result = self.process_single_image(path)
                results.append(result)
                print(f"Processed: {path} ({result.confidence:.1f}% confidence)")
            except Exception as e:
                logging.error(f"Failed to process {path}: {e}")
        
        return results


class OutputManager:
    """Handle various output formats"""
    
    @staticmethod
    def save_as_text(results: List[OCRResult], output_path: str):
        """Save results as plain text"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                f.write(f"=== Image {i + 1}: {result.image_path} ===\n")
                f.write(f"Engine: {result.engine}, Language: {result.language}\n")
                f.write(f"Confidence: {result.confidence:.1f}%\n")
                f.write(f"Processing Time: {result.processing_time:.2f}s\n")
                f.write(f"Timestamp: {result.timestamp}\n\n")
                f.write(result.text)
                f.write("\n\n" + "="*50 + "\n\n")
    
    @staticmethod
    def save_as_json(results: List[OCRResult], output_path: str):
        """Save results as JSON"""
        json_data = [asdict(result) for result in results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def save_as_csv(results: List[OCRResult], output_path: str):
        """Save results as CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'engine', 'language', 'confidence', 
                           'processing_time', 'timestamp', 'text'])
            
            for result in results:
                writer.writerow([
                    result.image_path, result.engine, result.language,
                    result.confidence, result.processing_time,
                    result.timestamp, result.text.replace('\n', '\\n')
                ])


def main():
    parser = argparse.ArgumentParser(
        description='ImageToText: Extract text from images using OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png                    # Extract text to stdout
  %(prog)s *.jpg -o output.txt          # Batch process to text file
  %(prog)s document.pdf --engine easyocr # Use EasyOCR engine
  %(prog)s image.png --format json      # Output as JSON
  %(prog)s image.png --language fra     # French OCR
        """
    )
    
    parser.add_argument('input_files', nargs='+',
                       help='Input image files or directories')
    parser.add_argument('-o', '--output', type=str,
                       help='Output file (default: stdout)')
    parser.add_argument('--engine', choices=['tesseract', 'easyocr'],
                       default='tesseract',
                       help='OCR engine to use (default: tesseract)')
    parser.add_argument('--language', type=str, default='eng',
                       help='Language for OCR (default: eng)')
    parser.add_argument('--format', choices=['text', 'json', 'csv'],
                       default='text',
                       help='Output format (default: text)')
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                       help='Minimum confidence threshold (0-100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--list-languages', action='store_true',
                       help='List available languages and exit')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # List available languages
    if args.list_languages:
        if args.engine == 'tesseract' and TESSERACT_AVAILABLE:
            try:
                langs = pytesseract.get_languages()
                print("Available Tesseract languages:")
                for lang in sorted(langs):
                    print(f"  {lang}")
            except:
                print("Could not retrieve Tesseract languages")
        elif args.engine == 'easyocr' and EASYOCR_AVAILABLE:
            print("Common EasyOCR language codes:")
            common_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
            for lang in common_langs:
                print(f"  {lang}")
        return
    
    # Check if required OCR engine is available
    if args.engine == 'tesseract' and not TESSERACT_AVAILABLE:
        print("Error: Tesseract OCR not available. Install with: pip install pytesseract pillow")
        sys.exit(1)
    elif args.engine == 'easyocr' and not EASYOCR_AVAILABLE:
        print("Error: EasyOCR not available. Install with: pip install easyocr")
        sys.exit(1)
    
    # Expand input files (handle wildcards and directories)
    input_files = []
    for pattern in args.input_files:
        path = Path(pattern)
        if path.is_dir():
            # Add all supported image files in directory
            for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.pdf']:
                input_files.extend(path.glob(f'*{ext}'))
                input_files.extend(path.glob(f'*{ext.upper()}'))
        elif path.exists():
            input_files.append(str(path))
        else:
            # Handle wildcards
            import glob
            matches = glob.glob(pattern)
            input_files.extend(matches)
    
    if not input_files:
        print("Error: No valid input files found")
        sys.exit(1)
    
    # Process files
    converter = ImageToTextConverter(args.engine, args.language)
    results = converter.process_batch([str(f) for f in input_files])
    
    # Filter by confidence threshold
    if args.confidence_threshold > 0:
        results = [r for r in results if r.confidence >= args.confidence_threshold]
        print(f"Filtered to {len(results)} results above {args.confidence_threshold}% confidence")
    
    if not results:
        print("No text extracted from input files")
        return
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        
        if args.format == 'text':
            OutputManager.save_as_text(results, str(output_path))
        elif args.format == 'json':
            OutputManager.save_as_json(results, str(output_path))
        elif args.format == 'csv':
            OutputManager.save_as_csv(results, str(output_path))
        
        print(f"Results saved to: {output_path}")
    else:
        # Print to stdout
        for i, result in enumerate(results):
            if len(results) > 1:
                print(f"\n=== {result.image_path} ===")
                print(f"Confidence: {result.confidence:.1f}%, Time: {result.processing_time:.2f}s")
                print("-" * 40)
            print(result.text)


if __name__ == '__main__':
    main()
