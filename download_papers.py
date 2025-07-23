#!/usr/bin/env python3
"""
Download papers script for DocuMentor AI
Usage: python download_papers.py [--max-papers 200] [--categories cs.LG,cs.CV]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.arxiv_downloader import ArxivDownloader


def main():
    parser = argparse.ArgumentParser(description="Download papers from arXiv")
    parser.add_argument(
        "--max-papers", 
        type=int, 
        default=200, 
        help="Maximum number of papers to download (default: 200)"
    )
    parser.add_argument(
        "--categories", 
        type=str, 
        default="cs.LG,cs.CV,cs.AI",
        help="Comma-separated list of arXiv categories (default: cs.LG,cs.CV,cs.AI)"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=1.0,
        help="Delay between downloads in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--days-back", 
        type=int, 
        default=730,
        help="How many days back to search (default: 730 = 2 years)"
    )
    
    args = parser.parse_args()
    
    # Parse categories
    categories = [cat.strip() for cat in args.categories.split(",")]
    
    # Calculate start date
    from datetime import timedelta
    start_date = datetime.now() - timedelta(days=args.days_back)
    
    print(f"🤖 DocuMentor AI - Paper Download")
    print(f"📊 Target papers: {args.max_papers}")
    print(f"🏷️ Categories: {', '.join(categories)}")
    print(f"📅 From date: {start_date.strftime('%Y-%m-%d')}")
    print(f"⏱️ Delay: {args.delay}s between downloads")
    print("=" * 50)
    
    # Initialize and run downloader
    downloader = ArxivDownloader()
    
    try:
        successful, failed = downloader.download_papers(
            categories=categories,
            max_papers=args.max_papers,
            start_date=start_date,
            delay_between_downloads=args.delay
        )
        
        # Show final statistics
        stats = downloader.get_download_statistics()
        print(f"\n🎉 Download Session Complete!")
        print(f"✅ Successfully downloaded: {successful} papers")
        print(f"❌ Failed downloads: {failed}")
        print(f"📁 Total papers in collection: {stats['total_papers']}")
        print(f"💾 Total storage used: {stats['total_size_gb']} GB")
        print(f"📂 Files location: {stats['download_directory']}")
        
        if stats['categories']:
            print(f"\n📊 Papers by category:")
            for cat, count in stats['categories'].items():
                print(f"   {cat}: {count} papers")
        
    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
        stats = downloader.get_download_statistics()
        print(f"📊 Current collection: {stats['total_papers']} papers")
    except Exception as e:
        print(f"💥 Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 