"""
arXiv Paper Downloader for DocuMentor AI
Downloads research papers from arXiv API with metadata extraction and progress tracking.
"""

import os
import sys
import time
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import arxiv
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


@dataclass
class PaperMetadata:
    """Dataclass to store paper metadata"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    pdf_url: str
    entry_id: str
    pdf_path: Optional[str] = None
    download_date: Optional[str] = None
    file_size_mb: Optional[float] = None
    page_count: Optional[int] = None
    md5_hash: Optional[str] = None


class ArxivDownloader:
    """Downloads papers from arXiv with robust error handling and progress tracking"""
    
    def __init__(self, download_dir: Path = None, metadata_file: str = "paper_metadata.json"):
        self.download_dir = download_dir or config.RAW_DATA_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.download_dir / metadata_file
        self.client = arxiv.Client()
        
        # Load existing metadata if available
        self.metadata = self._load_existing_metadata()
        
        print(f"📁 Download directory: {self.download_dir}")
        print(f"📊 Existing papers: {len(self.metadata)}")
    
    def _load_existing_metadata(self) -> Dict[str, PaperMetadata]:
        """Load existing metadata from JSON file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                return {k: PaperMetadata(**v) for k, v in data.items()}
            except Exception as e:
                print(f"⚠️ Error loading metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            data = {k: asdict(v) for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error saving metadata: {e}")
    
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        return file_path.stat().st_size / (1024 * 1024)
    
    def search_papers(self, 
                     categories: List[str], 
                     max_results: int = 200,
                     start_date: Optional[datetime] = None,
                     keywords: Optional[List[str]] = None) -> List[arxiv.Result]:
        """Search for papers on arXiv"""
        
        # Build search query
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        query = f"({category_query})"
        
        # Add date filter if specified
        if start_date:
            # Note: arXiv doesn't support date filtering in queries, we'll filter after
            pass
        
        # Add keywords if specified
        if keywords:
            keyword_query = " OR ".join(keywords)
            query += f" AND ({keyword_query})"
        
        print(f"🔍 Searching arXiv with query: {query}")
        print(f"📊 Requesting {max_results} papers...")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,  # Get extra to account for filtering
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = list(self.client.results(search))
        
        # Filter by date if specified
        if start_date:
            # Make start_date timezone-aware to match arXiv timestamps
            if start_date.tzinfo is None:
                import pytz
                start_date = pytz.UTC.localize(start_date)
            results = [r for r in results if r.published >= start_date]
        
        # Limit to requested number
        results = results[:max_results]
        
        print(f"✅ Found {len(results)} papers")
        return results
    
    def download_paper(self, result: arxiv.Result, max_retries: int = 3) -> Optional[PaperMetadata]:
        """Download a single paper with retry logic"""
        
        arxiv_id = result.get_short_id()
        
        # Skip if already downloaded
        if arxiv_id in self.metadata and self.metadata[arxiv_id].pdf_path:
            existing_path = Path(self.metadata[arxiv_id].pdf_path)
            if existing_path.exists():
                return self.metadata[arxiv_id]
        
        # Create filename
        safe_title = "".join(c for c in result.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title[:100]  # Limit length
        filename = f"{arxiv_id}_{safe_title}.pdf"
        file_path = self.download_dir / filename
        
        # Download with retries
        for attempt in range(max_retries):
            try:
                print(f"📥 Downloading: {result.title[:60]}...")
                
                response = requests.get(result.pdf_url, timeout=30)
                response.raise_for_status()
                
                # Save PDF
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # Create metadata
                metadata = PaperMetadata(
                    arxiv_id=arxiv_id,
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    categories=result.categories,
                    published=result.published.isoformat(),
                    pdf_url=result.pdf_url,
                    entry_id=result.entry_id,
                    pdf_path=str(file_path),
                    download_date=datetime.now().isoformat(),
                    file_size_mb=self._get_file_size_mb(file_path),
                    md5_hash=self._calculate_md5(file_path)
                )
                
                self.metadata[arxiv_id] = metadata
                return metadata
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"💥 Failed to download {arxiv_id} after {max_retries} attempts")
        
        return None
    
    def download_papers(self, 
                       categories: List[str] = None,
                       max_papers: int = 200,
                       start_date: Optional[datetime] = None,
                       keywords: Optional[List[str]] = None,
                       delay_between_downloads: float = 1.0) -> Tuple[int, int]:
        """Download multiple papers with progress tracking"""
        
        categories = categories or config.ARXIV_CONFIG['subjects']
        start_date = start_date or datetime.now() - timedelta(days=730)  # Last 2 years
        
        print(f"🚀 Starting download of {max_papers} papers")
        print(f"📅 From date: {start_date.strftime('%Y-%m-%d')}")
        print(f"🏷️ Categories: {', '.join(categories)}")
        
        # Search for papers
        papers = self.search_papers(
            categories=categories,
            max_results=max_papers,
            start_date=start_date,
            keywords=keywords
        )
        
        # Download papers
        successful_downloads = 0
        failed_downloads = 0
        
        with tqdm(total=len(papers), desc="Downloading papers") as pbar:
            for paper in papers:
                try:
                    metadata = self.download_paper(paper)
                    if metadata:
                        successful_downloads += 1
                        pbar.set_postfix({
                            'Success': successful_downloads,
                            'Failed': failed_downloads,
                            'Current': paper.title[:30] + '...'
                        })
                    else:
                        failed_downloads += 1
                    
                    # Save metadata periodically
                    if (successful_downloads + failed_downloads) % 10 == 0:
                        self._save_metadata()
                    
                    # Delay to be respectful to arXiv servers
                    time.sleep(delay_between_downloads)
                    
                except KeyboardInterrupt:
                    print("\n⚠️ Download interrupted by user")
                    break
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                    failed_downloads += 1
                
                pbar.update(1)
        
        # Save final metadata
        self._save_metadata()
        
        print(f"\n🎉 Download complete!")
        print(f"✅ Successful downloads: {successful_downloads}")
        print(f"❌ Failed downloads: {failed_downloads}")
        print(f"📁 Files saved to: {self.download_dir}")
        
        return successful_downloads, failed_downloads
    
    def get_download_statistics(self) -> Dict:
        """Get statistics about downloaded papers"""
        if not self.metadata:
            return {"total_papers": 0}
        
        papers = list(self.metadata.values())
        downloaded_papers = [p for p in papers if p.pdf_path and Path(p.pdf_path).exists()]
        
        total_size_mb = sum(p.file_size_mb or 0 for p in downloaded_papers)
        categories = {}
        for paper in downloaded_papers:
            for cat in paper.categories:
                categories[cat] = categories.get(cat, 0) + 1
        
        authors = []
        for paper in downloaded_papers:
            authors.extend(paper.authors)
        unique_authors = len(set(authors))
        
        return {
            "total_papers": len(downloaded_papers),
            "total_size_mb": round(total_size_mb, 2),
            "total_size_gb": round(total_size_mb / 1024, 2),
            "categories": categories,
            "unique_authors": unique_authors,
            "average_file_size_mb": round(total_size_mb / len(downloaded_papers), 2) if downloaded_papers else 0,
            "download_directory": str(self.download_dir),
            "metadata_file": str(self.metadata_file)
        }


def main():
    """Main function to run the downloader"""
    print("🤖 DocuMentor AI - arXiv Paper Downloader")
    print("=" * 50)
    
    # Initialize downloader
    downloader = ArxivDownloader()
    
    # Download papers
    successful, failed = downloader.download_papers(
        max_papers=200,  # Start with 200 papers
        delay_between_downloads=1.0  # 1 second delay between downloads
    )
    
    # Show statistics
    stats = downloader.get_download_statistics()
    print(f"\n📊 Download Statistics:")
    print(f"   Total papers: {stats['total_papers']}")
    print(f"   Total size: {stats['total_size_gb']} GB")
    print(f"   Categories: {stats['categories']}")
    print(f"   Average file size: {stats['average_file_size_mb']} MB")


if __name__ == "__main__":
    main() 