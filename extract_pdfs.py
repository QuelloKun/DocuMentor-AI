#!/usr/bin/env python3
"""
Extract text from PDFs script for DocuMentor AI
Usage: python extract_pdfs.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.pdf_extractor import PDFExtractor
import config


def main():
    print("🤖 DocuMentor AI - PDF Text Extraction")
    print("=" * 50)
    
    # Initialize extractor
    extractor = PDFExtractor()
    
    # Load metadata file
    metadata_file = config.RAW_DATA_DIR / "paper_metadata.json"
    
    print(f"📁 Processing PDFs from: {config.RAW_DATA_DIR}")
    print(f"📁 Saving extracted text to: {config.PROCESSED_DATA_DIR}")
    
    # Extract all PDFs
    extracted_papers = extractor.extract_all_pdfs(metadata_file)
    
    if extracted_papers:
        # Save extracted texts
        output_file = extractor.save_extracted_texts(extracted_papers)
        
        # Show detailed statistics
        stats = extractor.get_extraction_statistics(extracted_papers)
        
        print(f"\n🎉 Extraction Complete!")
        print(f"✅ Successfully processed: {stats['total_papers']} papers")
        print(f"📊 Total words extracted: {stats['total_words']:,}")
        print(f"📄 Average words per paper: {stats['average_words_per_paper']:,}")
        print(f"📝 Average sections per paper: {stats['average_sections_per_paper']:.1f}")
        print(f"⭐ Average quality score: {stats['average_quality_score']:.3f}")
        print(f"📋 Papers with abstracts: {stats['papers_with_abstract']} ({stats['abstract_coverage']:.1%})")
        print(f"📚 Papers with references: {stats['papers_with_references']} ({stats['reference_coverage']:.1%})")
        print(f"💾 Output saved to: {output_file}")
        
        # Quality assessment
        if stats['average_quality_score'] > 0.7:
            print(f"🟢 Extraction quality: Excellent")
        elif stats['average_quality_score'] > 0.5:
            print(f"🟡 Extraction quality: Good")
        else:
            print(f"🔴 Extraction quality: Needs improvement")
        
        # Next steps
        print(f"\n🚀 Ready for next phase: Fine-tuning dataset creation")
        
    else:
        print("❌ No papers were successfully extracted")
        print("🔍 Check PDF files and try again")


if __name__ == "__main__":
    main() 