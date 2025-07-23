"""
PDF Text Extractor for DocuMentor AI
Extracts clean, structured text from research papers using PyMuPDF.
"""

import os
import sys
import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


@dataclass
class ExtractedSection:
    """Represents a section of text extracted from a paper"""
    title: str
    content: str
    page_numbers: List[int]
    confidence: float  # How confident we are this is a real section


@dataclass
class PaperText:
    """Represents the extracted text content of a research paper"""
    arxiv_id: str
    title: str
    abstract: str
    sections: List[ExtractedSection]
    full_text: str
    page_count: int
    word_count: int
    char_count: int
    extraction_date: str
    quality_score: float  # Overall quality of extraction (0-1)
    has_references: bool
    language: str = "en"


class PDFExtractor:
    """Extracts structured text from research paper PDFs"""
    
    def __init__(self, input_dir: Path = None, output_dir: Path = None):
        self.input_dir = input_dir or config.RAW_DATA_DIR
        self.output_dir = output_dir or config.PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Common section headers in research papers
        self.section_patterns = [
            r'^\s*(\d+\.?\s*)?(abstract|introduction|related work|methodology|method|methods|approach|model|experiments?|evaluation|results?|discussion|conclusion|conclusions|future work|acknowledgments?|references|bibliography)\s*$',
            r'^\s*(\d+\.?\s*)?(background|literature review|problem statement|proposed method|implementation|dataset|analysis|limitations|comparison)\s*$',
        ]
        
        # Patterns to identify section headers
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.section_patterns]
        
        print(f"📁 Input directory: {self.input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'^\d+\s+', '', text, flags=re.MULTILINE)  # Page numbers
        text = re.sub(r'\f', ' ', text)  # Form feeds
        
        # Fix common OCR/extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenated words split across lines
        
        # Remove URLs and email addresses (often noise in academic papers)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Clean up and normalize
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def detect_sections(self, text: str) -> List[Tuple[str, int]]:
        """Detect section headers in the text"""
        lines = text.split('\n')
        sections = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches section pattern
            for pattern in self.compiled_patterns:
                if pattern.match(line):
                    # Calculate confidence based on line characteristics
                    confidence = 0.5
                    
                    # Boost confidence for common sections
                    if any(keyword in line.lower() for keyword in ['abstract', 'introduction', 'conclusion', 'method', 'result']):
                        confidence += 0.3
                    
                    # Boost if line is short (typical for headers)
                    if len(line.split()) <= 5:
                        confidence += 0.2
                    
                    # Boost if line is all caps or title case
                    if line.isupper() or line.istitle():
                        confidence += 0.1
                    
                    sections.append((line, i, min(confidence, 1.0)))
                    break
        
        return sections
    
    def extract_sections(self, text: str) -> List[ExtractedSection]:
        """Extract structured sections from text"""
        sections = self.detect_sections(text)
        extracted_sections = []
        
        lines = text.split('\n')
        
        for i, (section_title, start_line, confidence) in enumerate(sections):
            # Determine end line (next section or end of text)
            if i + 1 < len(sections):
                end_line = sections[i + 1][1]
            else:
                end_line = len(lines)
            
            # Extract section content
            section_lines = lines[start_line + 1:end_line]
            content = '\n'.join(section_lines).strip()
            content = self.clean_text(content)
            
            # Skip empty sections
            if not content or len(content.split()) < 10:
                continue
            
            extracted_section = ExtractedSection(
                title=section_title.strip(),
                content=content,
                page_numbers=[],  # Would need page tracking for this
                confidence=confidence
            )
            
            extracted_sections.append(extracted_section)
        
        return extracted_sections
    
    def extract_abstract(self, text: str) -> str:
        """Extract abstract from the beginning of the paper"""
        # Look for abstract section
        abstract_patterns = [
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:keywords|introduction|\d+\.\s*introduction|\d+\s+introduction))',
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*[A-Z][A-Za-z\s]{5,}:)',
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\n)'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                abstract = self.clean_text(abstract)
                if len(abstract.split()) > 20:  # Reasonable abstract length
                    return abstract
        
        # Fallback: take first paragraph that looks like an abstract
        paragraphs = text.split('\n\n')
        for para in paragraphs[:5]:
            para = para.strip()
            if (len(para.split()) > 50 and 
                not para.lower().startswith(('keywords', 'introduction', 'figure', 'table'))):
                return self.clean_text(para)
        
        return ""
    
    def calculate_quality_score(self, paper_text: PaperText) -> float:
        """Calculate quality score for extracted text (0-1)"""
        score = 0.0
        
        # Check if we have basic sections
        if paper_text.abstract:
            score += 0.2
        
        if len(paper_text.sections) >= 3:
            score += 0.3
        
        # Check for common academic sections
        section_titles = [s.title.lower() for s in paper_text.sections]
        required_sections = ['introduction', 'method', 'result', 'conclusion']
        found_sections = sum(1 for req in required_sections if any(req in title for title in section_titles))
        score += (found_sections / len(required_sections)) * 0.3
        
        # Check text quality
        if paper_text.word_count > 1000:
            score += 0.1
        
        if paper_text.has_references:
            score += 0.1
        
        return min(score, 1.0)
    
    def extract_pdf(self, pdf_path: Path, arxiv_id: str = None) -> Optional[PaperText]:
        """Extract text from a single PDF file"""
        try:
            if not pdf_path.exists():
                print(f"❌ PDF not found: {pdf_path}")
                return None
            
            # Extract arxiv_id from filename if not provided
            if not arxiv_id:
                arxiv_id = pdf_path.stem.split('_')[0]
            
            print(f"📄 Extracting: {arxiv_id}")
            
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                page_texts.append(page_text)
                full_text += page_text + "\n"
            
            doc.close()
            
            # Clean full text
            full_text = self.clean_text(full_text)
            
            if not full_text or len(full_text.split()) < 100:
                print(f"⚠️ Insufficient text extracted from {arxiv_id}")
                return None
            
            # Extract components
            abstract = self.extract_abstract(full_text)
            sections = self.extract_sections(full_text)
            
            # Detect if paper has references
            has_references = bool(re.search(r'references|bibliography', full_text, re.IGNORECASE))
            
            # Create paper text object
            paper_text = PaperText(
                arxiv_id=arxiv_id,
                title="",  # Will be filled from metadata if available
                abstract=abstract,
                sections=sections,
                full_text=full_text,
                page_count=len(page_texts),
                word_count=len(full_text.split()),
                char_count=len(full_text),
                extraction_date=datetime.now().isoformat(),
                quality_score=0.0,  # Will be calculated
                has_references=has_references
            )
            
            # Calculate quality score
            paper_text.quality_score = self.calculate_quality_score(paper_text)
            
            return paper_text
            
        except Exception as e:
            print(f"❌ Error extracting {pdf_path}: {e}")
            return None
    
    def extract_all_pdfs(self, metadata_file: Path = None) -> Dict[str, PaperText]:
        """Extract text from all PDFs in the input directory"""
        # Load existing metadata for titles
        metadata = {}
        if metadata_file and metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load metadata: {e}")
        
        # Find all PDF files
        pdf_files = list(self.input_dir.glob("*.pdf"))
        print(f"📊 Found {len(pdf_files)} PDF files to process")
        
        extracted_papers = {}
        failed_extractions = []
        
        # Process each PDF
        with tqdm(total=len(pdf_files), desc="Extracting PDFs") as pbar:
            for pdf_path in pdf_files:
                try:
                    # Extract arxiv ID from filename
                    arxiv_id = pdf_path.stem.split('_')[0]
                    
                    # Extract text
                    paper_text = self.extract_pdf(pdf_path, arxiv_id)
                    
                    if paper_text:
                        # Fill in title from metadata if available
                        if arxiv_id in metadata:
                            paper_text.title = metadata[arxiv_id].get('title', '')
                        
                        extracted_papers[arxiv_id] = paper_text
                        
                        pbar.set_postfix({
                            'Success': len(extracted_papers),
                            'Failed': len(failed_extractions),
                            'Quality': f"{paper_text.quality_score:.2f}",
                            'Words': paper_text.word_count
                        })
                    else:
                        failed_extractions.append(arxiv_id)
                
                except Exception as e:
                    print(f"❌ Failed to process {pdf_path}: {e}")
                    failed_extractions.append(pdf_path.stem)
                
                pbar.update(1)
        
        print(f"\n🎉 Extraction complete!")
        print(f"✅ Successfully extracted: {len(extracted_papers)} papers")
        print(f"❌ Failed extractions: {len(failed_extractions)}")
        
        return extracted_papers
    
    def save_extracted_texts(self, extracted_papers: Dict[str, PaperText], 
                           filename: str = "extracted_papers.json") -> Path:
        """Save extracted texts to JSON file"""
        output_file = self.output_dir / filename
        
        # Convert to serializable format
        data = {}
        for arxiv_id, paper_text in extracted_papers.items():
            data[arxiv_id] = asdict(paper_text)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved extracted texts to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error saving extracted texts: {e}")
            return None
    
    def get_extraction_statistics(self, extracted_papers: Dict[str, PaperText]) -> Dict:
        """Get statistics about extracted papers"""
        if not extracted_papers:
            return {"total_papers": 0}
        
        papers = list(extracted_papers.values())
        
        total_words = sum(p.word_count for p in papers)
        total_chars = sum(p.char_count for p in papers)
        total_sections = sum(len(p.sections) for p in papers)
        
        quality_scores = [p.quality_score for p in papers]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        papers_with_abstract = sum(1 for p in papers if p.abstract)
        papers_with_references = sum(1 for p in papers if p.has_references)
        
        return {
            "total_papers": len(papers),
            "total_words": total_words,
            "total_characters": total_chars,
            "total_sections": total_sections,
            "average_words_per_paper": total_words // len(papers),
            "average_sections_per_paper": total_sections / len(papers),
            "average_quality_score": round(avg_quality, 3),
            "papers_with_abstract": papers_with_abstract,
            "papers_with_references": papers_with_references,
            "abstract_coverage": round(papers_with_abstract / len(papers), 3),
            "reference_coverage": round(papers_with_references / len(papers), 3),
            "output_directory": str(self.output_dir)
        }


def main():
    """Main function to run the PDF extractor"""
    print("🤖 DocuMentor AI - PDF Text Extractor")
    print("=" * 50)
    
    # Initialize extractor
    extractor = PDFExtractor()
    
    # Load metadata file
    metadata_file = config.RAW_DATA_DIR / "paper_metadata.json"
    
    # Extract all PDFs
    extracted_papers = extractor.extract_all_pdfs(metadata_file)
    
    # Save extracted texts
    if extracted_papers:
        output_file = extractor.save_extracted_texts(extracted_papers)
        
        # Show statistics
        stats = extractor.get_extraction_statistics(extracted_papers)
        print(f"\n📊 Extraction Statistics:")
        print(f"   Total papers processed: {stats['total_papers']}")
        print(f"   Total words extracted: {stats['total_words']:,}")
        print(f"   Average words per paper: {stats['average_words_per_paper']:,}")
        print(f"   Average sections per paper: {stats['average_sections_per_paper']:.1f}")
        print(f"   Average quality score: {stats['average_quality_score']}")
        print(f"   Papers with abstracts: {stats['papers_with_abstract']} ({stats['abstract_coverage']:.1%})")
        print(f"   Papers with references: {stats['papers_with_references']} ({stats['reference_coverage']:.1%})")
        print(f"   Output saved to: {output_file}")
    else:
        print("❌ No papers were successfully extracted")


if __name__ == "__main__":
    main() 