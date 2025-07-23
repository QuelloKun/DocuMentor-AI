#!/usr/bin/env python3
"""
Monitor download progress for DocuMentor AI
Shows current statistics and progress
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.arxiv_downloader import ArxivDownloader


def main():
    print("📊 DocuMentor AI - Download Progress Monitor")
    print("=" * 50)
    
    # Initialize downloader to access existing data
    downloader = ArxivDownloader()
    stats = downloader.get_download_statistics()
    
    if stats['total_papers'] == 0:
        print("🔍 No papers downloaded yet.")
        return
    
    # Show current statistics
    print(f"📁 Total papers downloaded: {stats['total_papers']}")
    print(f"💾 Total storage used: {stats['total_size_gb']} GB ({stats['total_size_mb']} MB)")
    print(f"📄 Average file size: {stats['average_file_size_mb']} MB")
    print(f"👥 Unique authors: {stats['unique_authors']}")
    print(f"📂 Download directory: {stats['download_directory']}")
    
    if stats['categories']:
        print(f"\n📊 Papers by category:")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat}: {count} papers")
    
    # Show recent downloads
    metadata_file = Path(stats['download_directory']) / "paper_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Sort by download date
            papers = []
            for paper_data in metadata.values():
                if paper_data.get('download_date'):
                    papers.append(paper_data)
            
            papers.sort(key=lambda x: x['download_date'], reverse=True)
            
            print(f"\n🕒 Recent downloads (last 5):")
            for i, paper in enumerate(papers[:5], 1):
                download_time = datetime.fromisoformat(paper['download_date']).strftime('%H:%M:%S')
                title = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
                print(f"   {i}. [{download_time}] {title}")
                
        except Exception as e:
            print(f"⚠️ Error reading metadata: {e}")
    
    # Check for PDF files in directory
    pdf_files = list(Path(stats['download_directory']).glob("*.pdf"))
    print(f"\n📁 PDF files in directory: {len(pdf_files)}")
    
    if len(pdf_files) != stats['total_papers']:
        print(f"⚠️ Mismatch: {stats['total_papers']} in metadata vs {len(pdf_files)} PDF files")
    
    print(f"\n✅ Collection Status: {stats['total_papers']} papers ready for processing")


if __name__ == "__main__":
    main() 