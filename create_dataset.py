#!/usr/bin/env python3
"""
Create fine-tuning dataset script for DocuMentor AI
Usage: python create_dataset.py [--max-papers 100] [--target-qa 1000]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.dataset_creator import DatasetCreator


def main():
    parser = argparse.ArgumentParser(description="Create fine-tuning dataset from extracted papers")
    parser.add_argument(
        "--max-papers", 
        type=int, 
        default=None, 
        help="Maximum number of papers to process (default: all)"
    )
    parser.add_argument(
        "--target-qa", 
        type=int, 
        default=1000,
        help="Target number of Q&A pairs to generate (default: 1000)"
    )
    parser.add_argument(
        "--formats", 
        type=str, 
        default="json,jsonl,csv",
        help="Output formats (comma-separated: json,jsonl,csv)"
    )
    
    args = parser.parse_args()
    
    # Parse formats
    formats = [fmt.strip() for fmt in args.formats.split(",")]
    
    print(f"🤖 DocuMentor AI - Dataset Creation")
    print(f"📊 Max papers: {args.max_papers or 'All'}")
    print(f"🎯 Target Q&A pairs: {args.target_qa}")
    print(f"📁 Output formats: {', '.join(formats)}")
    print("=" * 50)
    
    # Initialize creator
    creator = DatasetCreator()
    
    try:
        # Create dataset
        qa_pairs = creator.create_dataset(
            max_papers=args.max_papers,
            target_qa_pairs=args.target_qa
        )
        
        if qa_pairs:
            # Save in requested formats
            saved_files = creator.save_dataset(qa_pairs, formats=formats)
            
            # Show detailed statistics
            stats = creator.get_dataset_statistics(qa_pairs)
            
            print(f"\n🎉 Dataset Creation Complete!")
            print(f"✅ Generated {stats['total_pairs']} Q&A pairs")
            print(f"📚 From {stats['unique_papers']} unique papers")
            print(f"📄 Average {stats['avg_questions_per_paper']:.1f} questions per paper")
            print(f"💬 Average question: {stats['avg_question_length']:.1f} words")
            print(f"💭 Average answer: {stats['avg_answer_length']:.1f} words")
            
            print(f"\n📊 Content Distribution:")
            for qa_type, count in stats['type_distribution'].items():
                percentage = (count / stats['total_pairs']) * 100
                print(f"   {qa_type}: {count} pairs ({percentage:.1f}%)")
            
            print(f"\n🎯 Difficulty Distribution:")
            for difficulty, count in stats['difficulty_distribution'].items():
                percentage = (count / stats['total_pairs']) * 100
                print(f"   {difficulty}: {count} pairs ({percentage:.1f}%)")
            
            print(f"\n💾 Files Created:")
            for file_path in saved_files:
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                print(f"   {file_path.name}: {file_size:.2f} MB")
            
            # Quality assessment
            if stats['avg_answer_length'] > 50:
                print(f"\n🟢 Quality Assessment: Excellent - Rich, detailed answers")
            elif stats['avg_answer_length'] > 25:
                print(f"\n🟡 Quality Assessment: Good - Adequate answer length")
            else:
                print(f"\n🔴 Quality Assessment: Poor - Short answers, may need improvement")
            
            # Next steps
            print(f"\n🚀 Next Steps:")
            print(f"   1. Review sample Q&A pairs for quality")
            print(f"   2. Use .jsonl file for model fine-tuning")
            print(f"   3. Proceed to Phase 2: Model Training")
            
        else:
            print("❌ No dataset created - check input files and try again")
            
    except KeyboardInterrupt:
        print("\n⚠️ Dataset creation interrupted by user")
    except Exception as e:
        print(f"💥 Error during dataset creation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 