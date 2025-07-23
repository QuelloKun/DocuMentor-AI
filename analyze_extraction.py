#!/usr/bin/env python3
"""
Analyze extracted paper data for DocuMentor AI
Shows statistics and quality metrics of the extraction process
"""

import json
import sys
from pathlib import Path
from collections import Counter

def load_extracted_data(file_path):
    """Load the extracted papers data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_extraction_quality(data):
    """Analyze the quality of text extraction"""
    print("📊 Extraction Quality Analysis")
    print("=" * 50)
    
    papers = list(data.values())
    
    # Basic statistics
    total_papers = len(papers)
    total_words = sum(p['word_count'] for p in papers)
    total_chars = sum(p['char_count'] for p in papers)
    
    print(f"📁 Total papers processed: {total_papers}")
    print(f"📝 Total words extracted: {total_words:,}")
    print(f"📄 Total characters: {total_chars:,}")
    print(f"📊 Average words per paper: {total_words // total_papers:,}")
    print(f"📋 Average characters per paper: {total_chars // total_papers:,}")
    
    # Quality metrics
    quality_scores = [p['quality_score'] for p in papers]
    avg_quality = sum(quality_scores) / len(quality_scores)
    
    abstracts_found = sum(1 for p in papers if p['abstract'])
    references_found = sum(1 for p in papers if p['has_references'])
    sections_found = sum(len(p['sections']) for p in papers)
    
    print(f"\n🎯 Quality Metrics:")
    print(f"   Average quality score: {avg_quality:.3f}")
    print(f"   Papers with abstracts: {abstracts_found} ({abstracts_found/total_papers:.1%})")
    print(f"   Papers with references: {references_found} ({references_found/total_papers:.1%})")
    print(f"   Total sections detected: {sections_found}")
    print(f"   Average sections per paper: {sections_found/total_papers:.1f}")
    
    # Word count distribution
    word_counts = [p['word_count'] for p in papers]
    word_counts.sort()
    
    print(f"\n📈 Word Count Distribution:")
    print(f"   Minimum: {min(word_counts):,} words")
    print(f"   25th percentile: {word_counts[len(word_counts)//4]:,} words")
    print(f"   Median: {word_counts[len(word_counts)//2]:,} words")
    print(f"   75th percentile: {word_counts[3*len(word_counts)//4]:,} words")
    print(f"   Maximum: {max(word_counts):,} words")
    
    # Show papers with different quality levels
    high_quality = [p for p in papers if p['quality_score'] >= 0.7]
    medium_quality = [p for p in papers if 0.4 <= p['quality_score'] < 0.7]
    low_quality = [p for p in papers if p['quality_score'] < 0.4]
    
    print(f"\n🏆 Quality Distribution:")
    print(f"   High quality (≥0.7): {len(high_quality)} papers")
    print(f"   Medium quality (0.4-0.7): {len(medium_quality)} papers")
    print(f"   Low quality (<0.4): {len(low_quality)} papers")
    
    return {
        'total_papers': total_papers,
        'total_words': total_words,
        'avg_quality': avg_quality,
        'abstracts_found': abstracts_found,
        'references_found': references_found,
        'sections_found': sections_found
    }

def show_sample_papers(data, num_samples=3):
    """Show sample papers for manual inspection"""
    print(f"\n🔍 Sample Papers (showing {num_samples}):")
    print("=" * 50)
    
    papers = list(data.items())
    
    for i, (paper_id, paper) in enumerate(papers[:num_samples]):
        print(f"\n📄 Paper {i+1}: {paper_id}")
        print(f"   Title: {paper['title'][:80]}{'...' if len(paper['title']) > 80 else ''}")
        print(f"   Words: {paper['word_count']:,}")
        print(f"   Quality: {paper['quality_score']:.3f}")
        print(f"   Sections: {len(paper['sections'])}")
        print(f"   Abstract: {'✅' if paper['abstract'] else '❌'}")
        print(f"   References: {'✅' if paper['has_references'] else '❌'}")
        
        if paper['abstract']:
            print(f"   Abstract preview: {paper['abstract'][:150]}...")
        
        # Show full text preview
        print(f"   Full text preview: {paper['full_text'][:200]}...")

def assess_readiness_for_fine_tuning(stats):
    """Assess if the extracted data is ready for fine-tuning dataset creation"""
    print(f"\n🎯 Fine-Tuning Readiness Assessment:")
    print("=" * 50)
    
    readiness_score = 0
    max_score = 5
    
    # Check 1: Sufficient papers
    if stats['total_papers'] >= 100:
        print("✅ Sufficient number of papers (≥100)")
        readiness_score += 1
    else:
        print(f"⚠️ Limited papers: {stats['total_papers']} (recommend ≥100)")
    
    # Check 2: Sufficient content
    if stats['total_words'] >= 500000:
        print("✅ Sufficient text content (≥500k words)")
        readiness_score += 1
    else:
        print(f"⚠️ Limited content: {stats['total_words']:,} words (recommend ≥500k)")
    
    # Check 3: Good abstract coverage
    abstract_rate = stats['abstracts_found'] / stats['total_papers']
    if abstract_rate >= 0.9:
        print("✅ Excellent abstract coverage (≥90%)")
        readiness_score += 1
    elif abstract_rate >= 0.7:
        print("🟡 Good abstract coverage (≥70%)")
        readiness_score += 0.5
    else:
        print(f"⚠️ Low abstract coverage: {abstract_rate:.1%}")
    
    # Check 4: Reference detection
    ref_rate = stats['references_found'] / stats['total_papers']
    if ref_rate >= 0.8:
        print("✅ Good reference detection (≥80%)")
        readiness_score += 1
    else:
        print(f"🟡 Reference detection: {ref_rate:.1%}")
        readiness_score += 0.5
    
    # Check 5: Quality scores
    if stats['avg_quality'] >= 0.6:
        print("✅ Good average quality score (≥0.6)")
        readiness_score += 1
    elif stats['avg_quality'] >= 0.4:
        print("🟡 Acceptable quality score (≥0.4)")
        readiness_score += 0.5
    else:
        print(f"⚠️ Low quality score: {stats['avg_quality']:.3f}")
    
    # Overall assessment
    print(f"\n📊 Overall Readiness Score: {readiness_score:.1f}/{max_score}")
    
    if readiness_score >= 4:
        print("🟢 READY: Excellent quality for fine-tuning dataset creation")
    elif readiness_score >= 3:
        print("🟡 READY: Good quality for fine-tuning dataset creation")
    elif readiness_score >= 2:
        print("🟠 CAUTION: Acceptable but could be improved")
    else:
        print("🔴 NOT READY: Quality improvements needed")
    
    return readiness_score

def main():
    """Main analysis function"""
    print("🤖 DocuMentor AI - Extraction Analysis")
    print("=" * 50)
    
    # Load data
    data_file = Path("data/processed/extracted_papers.json")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📁 Loading data from: {data_file}")
    data = load_extracted_data(data_file)
    
    if not data:
        print("❌ Failed to load data")
        return
    
    # Perform analysis
    stats = analyze_extraction_quality(data)
    show_sample_papers(data, num_samples=3)
    readiness_score = assess_readiness_for_fine_tuning(stats)
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print("=" * 30)
    
    if readiness_score >= 3:
        print("✅ Proceed to fine-tuning dataset creation")
        print("✅ Use GPT-4 or similar to generate Q&A pairs from abstracts and full text")
        print("✅ Focus on methodology, results, and technical details")
    else:
        print("⚠️ Consider improving extraction quality first")
        print("⚠️ Add more papers to increase diversity")
        print("⚠️ Manually review low-quality extractions")
    
    print(f"\n🎯 Next Phase: Create Q&A dataset from {stats['total_papers']} papers")

if __name__ == "__main__":
    main() 