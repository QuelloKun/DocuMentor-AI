"""
Fine-tuning Dataset Creator for DocuMentor AI
Generates high-quality Question/Answer pairs from extracted research papers.
"""

import os
import sys
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


@dataclass
class QAPair:
    """Represents a Question/Answer pair for fine-tuning"""
    question: str
    answer: str
    context: str  # Source text that inspired the Q&A
    paper_id: str
    paper_title: str
    qa_type: str  # 'abstract', 'methodology', 'results', 'general'
    difficulty: str  # 'easy', 'medium', 'hard'
    generated_date: str


class DatasetCreator:
    """Creates fine-tuning datasets from extracted research papers"""
    
    def __init__(self, input_file: Path = None, output_dir: Path = None):
        self.input_file = input_file or config.PROCESSED_DATA_DIR / "extracted_papers.json"
        self.output_dir = output_dir or config.PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Question templates for different types
        self.question_templates = {
            'abstract': [
                "What is the main contribution of this paper?",
                "What problem does this research address?",
                "What methodology is used in this study?",
                "What are the key findings of this research?",
                "How does this work differ from previous approaches?",
                "What is the scope of this research?",
                "What are the main objectives of this study?",
                "What gap in existing research does this paper fill?",
            ],
            'methodology': [
                "How was this study conducted?",
                "What experimental setup was used?",
                "What datasets were used in this research?",
                "What evaluation metrics were employed?",
                "How were the results validated?",
                "What are the implementation details?",
                "What preprocessing steps were taken?",
                "How was the model trained?",
            ],
            'results': [
                "What were the main experimental results?",
                "How does this approach perform compared to baselines?",
                "What are the quantitative results?",
                "What limitations were identified?",
                "What are the implications of these findings?",
                "How robust are the experimental results?",
                "What ablation studies were conducted?",
                "What is the computational complexity?",
            ],
            'general': [
                "What are the practical applications of this research?",
                "What future work is suggested?",
                "What are the broader implications?",
                "How might this work impact the field?",
                "What related work is discussed?",
                "What assumptions does this work make?",
                "What are the ethical considerations?",
                "How does this relate to current industry practices?",
            ]
        }
        
        print(f"📁 Input file: {self.input_file}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_extracted_papers(self) -> Dict:
        """Load the extracted papers data"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading papers: {e}")
            return {}
    
    def clean_text_for_qa(self, text: str) -> str:
        """Clean and prepare text for Q&A generation"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove citations and references patterns
        text = re.sub(r'\[[0-9,\s-]+\]', '', text)  # [1], [1,2], [1-5]
        text = re.sub(r'\(et al\.\s*,?\s*\d{4}\)', '', text)  # (et al., 2020)
        text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)  # (Author, 2020)
        
        # Clean up formatting artifacts
        text = re.sub(r'∗|†|‡|§|¶', '', text)  # Remove symbols
        text = re.sub(r'\s*\n\s*', ' ', text)  # Replace newlines with spaces
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        return text.strip()
    
    def extract_key_sentences(self, text: str, num_sentences: int = 3) -> List[str]:
        """Extract key sentences from text for Q&A generation"""
        if not text:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= num_sentences:
            return sentences
        
        # Score sentences based on key indicators
        scored_sentences = []
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Boost sentences with methodology keywords
            method_keywords = ['method', 'approach', 'algorithm', 'model', 'framework', 'technique']
            score += sum(2 for kw in method_keywords if kw in sentence_lower)
            
            # Boost sentences with result keywords
            result_keywords = ['result', 'performance', 'accuracy', 'improvement', 'achieve', 'demonstrate']
            score += sum(2 for kw in result_keywords if kw in sentence_lower)
            
            # Boost sentences with contribution keywords
            contrib_keywords = ['propose', 'introduce', 'present', 'novel', 'new', 'contribution']
            score += sum(3 for kw in contrib_keywords if kw in sentence_lower)
            
            # Prefer longer sentences (more information)
            score += len(sentence.split()) / 10
            
            # Avoid very short or very long sentences
            if len(sentence.split()) < 5 or len(sentence.split()) > 50:
                score -= 2
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_sentences[:num_sentences]]
    
    def generate_qa_from_abstract(self, paper: Dict) -> List[QAPair]:
        """Generate Q&A pairs from paper abstract"""
        abstract = self.clean_text_for_qa(paper.get('abstract', ''))
        if not abstract or len(abstract.split()) < 20:
            return []
        
        qa_pairs = []
        
        # Generate different types of questions about the abstract
        templates = self.question_templates['abstract']
        
        for i, template in enumerate(templates[:4]):  # Limit to 4 questions per abstract
            # Modify template to be more specific
            question = template
            
            # Use the abstract as the answer context
            answer = self.generate_answer_from_context(question, abstract, paper)
            
            if answer:
                qa_pair = QAPair(
                    question=question,
                    answer=answer,
                    context=abstract,
                    paper_id=paper.get('arxiv_id', ''),
                    paper_title=paper.get('title', ''),
                    qa_type='abstract',
                    difficulty=['easy', 'medium'][i % 2],
                    generated_date=datetime.now().isoformat()
                )
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def generate_qa_from_full_text(self, paper: Dict) -> List[QAPair]:
        """Generate Q&A pairs from full paper text"""
        full_text = self.clean_text_for_qa(paper.get('full_text', ''))
        if not full_text or len(full_text.split()) < 100:
            return []
        
        qa_pairs = []
        
        # Extract key sentences from different parts of the paper
        text_chunks = self.split_text_into_chunks(full_text, chunk_size=500)
        
        # Generate questions for each chunk type
        for chunk_idx, chunk in enumerate(text_chunks[:6]):  # Limit to 6 chunks
            chunk_type = self.classify_text_chunk(chunk)
            templates = self.question_templates.get(chunk_type, self.question_templates['general'])
            
            # Generate 1-2 questions per chunk
            for template in templates[:2]:
                question = template
                answer = self.generate_answer_from_context(question, chunk, paper)
                
                if answer:
                    qa_pair = QAPair(
                        question=question,
                        answer=answer,
                        context=chunk[:500] + "...",  # Truncate context
                        paper_id=paper.get('arxiv_id', ''),
                        paper_title=paper.get('title', ''),
                        qa_type=chunk_type,
                        difficulty=['easy', 'medium', 'hard'][chunk_idx % 3],
                        generated_date=datetime.now().isoformat()
                    )
                    qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into meaningful chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            
            # Try to end chunks at sentence boundaries
            if i + chunk_size < len(words):
                # Look for sentence ending in last 50 words
                last_part = ' '.join(chunk_words[-50:])
                sentence_end = max(
                    last_part.rfind('.'),
                    last_part.rfind('!'),
                    last_part.rfind('?')
                )
                
                if sentence_end > 0:
                    # Adjust chunk to end at sentence
                    cut_point = len(' '.join(chunk_words[:-50])) + sentence_end + 1
                    chunk = chunk[:cut_point]
            
            chunks.append(chunk)
        
        return chunks
    
    def classify_text_chunk(self, chunk: str) -> str:
        """Classify what type of content a text chunk contains"""
        chunk_lower = chunk.lower()
        
        # Check for methodology indicators
        method_keywords = ['method', 'approach', 'algorithm', 'model', 'implement', 'design', 'framework']
        method_score = sum(1 for kw in method_keywords if kw in chunk_lower)
        
        # Check for results indicators
        result_keywords = ['result', 'performance', 'accuracy', 'evaluation', 'experiment', 'test']
        result_score = sum(1 for kw in result_keywords if kw in chunk_lower)
        
        # Check for abstract indicators (usually at beginning)
        abstract_keywords = ['abstract', 'propose', 'introduce', 'present', 'contribution']
        abstract_score = sum(1 for kw in abstract_keywords if kw in chunk_lower)
        
        # Classify based on highest score
        scores = {
            'methodology': method_score,
            'results': result_score,
            'abstract': abstract_score
        }
        
        best_type = max(scores, key=scores.get)
        if scores[best_type] > 0:
            return best_type
        
        return 'general'
    
    def generate_answer_from_context(self, question: str, context: str, paper: Dict) -> str:
        """Generate an answer based on question and context"""
        # This is a simplified answer generation
        # In a real implementation, you might use a language model here
        
        question_lower = question.lower()
        context_sentences = re.split(r'[.!?]+', context)
        context_sentences = [s.strip() for s in context_sentences if len(s.strip()) > 10]
        
        # Score sentences based on relevance to question
        relevant_sentences = []
        
        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            relevance = 0
            
            # Check for keyword overlap
            question_words = set(re.findall(r'\b\w+\b', question_lower))
            sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
            
            overlap = len(question_words & sentence_words)
            relevance += overlap * 2
            
            # Boost based on question type
            if 'main contribution' in question_lower or 'contribution' in question_lower:
                if any(kw in sentence_lower for kw in ['propose', 'present', 'introduce', 'novel']):
                    relevance += 5
            
            if 'problem' in question_lower or 'address' in question_lower:
                if any(kw in sentence_lower for kw in ['problem', 'challenge', 'issue', 'limitation']):
                    relevance += 5
            
            if 'methodology' in question_lower or 'conducted' in question_lower:
                if any(kw in sentence_lower for kw in ['method', 'approach', 'algorithm', 'implement']):
                    relevance += 5
            
            if 'result' in question_lower or 'performance' in question_lower:
                if any(kw in sentence_lower for kw in ['result', 'performance', 'achieve', 'accuracy']):
                    relevance += 5
            
            if relevance > 0:
                relevant_sentences.append((sentence, relevance))
        
        # Sort by relevance and combine top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if not relevant_sentences:
            return ""
        
        # Combine top 2-3 relevant sentences
        answer_sentences = [s[0] for s in relevant_sentences[:3]]
        answer = '. '.join(answer_sentences)
        
        # Clean up the answer
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Ensure answer is substantial
        if len(answer.split()) < 10:
            return ""
        
        return answer
    
    def generate_comparative_questions(self, papers: List[Dict]) -> List[QAPair]:
        """Generate questions that compare different papers"""
        if len(papers) < 2:
            return []
        
        qa_pairs = []
        comparative_templates = [
            "How do the approaches in these papers differ?",
            "What are the key differences in methodology?",
            "Which approach shows better performance?",
            "How do the experimental setups compare?",
        ]
        
        # Sample pairs of papers
        for i in range(min(3, len(papers) - 1)):
            paper1 = papers[i]
            paper2 = papers[i + 1]
            
            for template in comparative_templates[:2]:  # Limit comparative questions
                context = f"Paper 1: {paper1.get('abstract', '')[:300]}...\n\nPaper 2: {paper2.get('abstract', '')[:300]}..."
                
                # Generate comparative answer
                answer = f"The first paper ({paper1.get('title', 'Paper 1')[:50]}...) focuses on {self.extract_key_focus(paper1)}, while the second paper ({paper2.get('title', 'Paper 2')[:50]}...) emphasizes {self.extract_key_focus(paper2)}."
                
                qa_pair = QAPair(
                    question=template,
                    answer=answer,
                    context=context,
                    paper_id=f"{paper1.get('arxiv_id', '')},{paper2.get('arxiv_id', '')}",
                    paper_title=f"Comparison: {paper1.get('title', '')[:30]}... vs {paper2.get('title', '')[:30]}...",
                    qa_type='comparison',
                    difficulty='hard',
                    generated_date=datetime.now().isoformat()
                )
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def extract_key_focus(self, paper: Dict) -> str:
        """Extract the key focus/contribution of a paper"""
        abstract = paper.get('abstract', '')
        
        # Look for key phrases
        focus_patterns = [
            r'propose[s]?\s+([^.]{20,80})',
            r'present[s]?\s+([^.]{20,80})',
            r'introduce[s]?\s+([^.]{20,80})',
            r'develop[s]?\s+([^.]{20,80})',
        ]
        
        for pattern in focus_patterns:
            match = re.search(pattern, abstract, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: use first meaningful sentence
        sentences = re.split(r'[.!?]+', abstract)
        for sentence in sentences:
            if len(sentence.split()) > 5:
                return sentence.strip()[:100]
        
        return "novel approaches"
    
    def create_dataset(self, max_papers: int = None, target_qa_pairs: int = 1000) -> List[QAPair]:
        """Create the complete fine-tuning dataset"""
        print("🤖 Creating Fine-tuning Dataset")
        print("=" * 50)
        
        # Load papers
        papers_data = self.load_extracted_papers()
        if not papers_data:
            print("❌ No papers loaded")
            return []
        
        papers = list(papers_data.values())
        if max_papers:
            papers = papers[:max_papers]
        
        print(f"📊 Processing {len(papers)} papers")
        print(f"🎯 Target: {target_qa_pairs} Q&A pairs")
        
        all_qa_pairs = []
        
        # Generate Q&A pairs from individual papers
        with tqdm(total=len(papers), desc="Generating Q&A pairs") as pbar:
            for paper in papers:
                paper_qa_pairs = []
                
                # Generate from abstract
                abstract_pairs = self.generate_qa_from_abstract(paper)
                paper_qa_pairs.extend(abstract_pairs)
                
                # Generate from full text
                full_text_pairs = self.generate_qa_from_full_text(paper)
                paper_qa_pairs.extend(full_text_pairs)
                
                all_qa_pairs.extend(paper_qa_pairs)
                
                pbar.set_postfix({
                    'Total QA': len(all_qa_pairs),
                    'This Paper': len(paper_qa_pairs),
                    'Progress': f"{len(all_qa_pairs)}/{target_qa_pairs}"
                })
                
                pbar.update(1)
                
                # Stop if we've reached target
                if len(all_qa_pairs) >= target_qa_pairs:
                    break
        
        # Generate comparative questions
        if len(papers) > 1:
            print("🔄 Generating comparative questions...")
            comparative_pairs = self.generate_comparative_questions(papers)
            all_qa_pairs.extend(comparative_pairs)
        
        # Shuffle and limit to target
        random.shuffle(all_qa_pairs)
        all_qa_pairs = all_qa_pairs[:target_qa_pairs]
        
        print(f"\n🎉 Dataset Creation Complete!")
        print(f"✅ Generated {len(all_qa_pairs)} Q&A pairs")
        
        return all_qa_pairs
    
    def save_dataset(self, qa_pairs: List[QAPair], formats: List[str] = ['json', 'jsonl', 'csv']):
        """Save dataset in multiple formats"""
        if not qa_pairs:
            print("❌ No Q&A pairs to save")
            return
        
        base_filename = f"documentor_qa_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert to serializable format
        qa_data = [asdict(qa) for qa in qa_pairs]
        
        saved_files = []
        
        # Save as JSON
        if 'json' in formats:
            json_file = self.output_dir / f"{base_filename}.json"
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(qa_data, f, indent=2, ensure_ascii=False)
                saved_files.append(json_file)
                print(f"💾 Saved JSON: {json_file}")
            except Exception as e:
                print(f"❌ Error saving JSON: {e}")
        
        # Save as JSONL (for training)
        if 'jsonl' in formats:
            jsonl_file = self.output_dir / f"{base_filename}.jsonl"
            try:
                with open(jsonl_file, 'w', encoding='utf-8') as f:
                    for qa in qa_data:
                        # Format for training: instruction-input-output
                        training_format = {
                            "instruction": "Answer the following question about the research paper:",
                            "input": qa['question'],
                            "output": qa['answer'],
                            "paper_id": qa['paper_id'],
                            "paper_title": qa['paper_title']
                        }
                        f.write(json.dumps(training_format, ensure_ascii=False) + '\n')
                saved_files.append(jsonl_file)
                print(f"💾 Saved JSONL: {jsonl_file}")
            except Exception as e:
                print(f"❌ Error saving JSONL: {e}")
        
        # Save as CSV
        if 'csv' in formats:
            csv_file = self.output_dir / f"{base_filename}.csv"
            try:
                df = pd.DataFrame(qa_data)
                df.to_csv(csv_file, index=False, encoding='utf-8')
                saved_files.append(csv_file)
                print(f"💾 Saved CSV: {csv_file}")
            except Exception as e:
                print(f"❌ Error saving CSV: {e}")
        
        return saved_files
    
    def get_dataset_statistics(self, qa_pairs: List[QAPair]) -> Dict:
        """Get statistics about the generated dataset"""
        if not qa_pairs:
            return {"total_pairs": 0}
        
        # Type distribution
        type_counts = {}
        difficulty_counts = {}
        paper_counts = {}
        
        total_question_length = 0
        total_answer_length = 0
        
        for qa in qa_pairs:
            # Count by type
            type_counts[qa.qa_type] = type_counts.get(qa.qa_type, 0) + 1
            
            # Count by difficulty
            difficulty_counts[qa.difficulty] = difficulty_counts.get(qa.difficulty, 0) + 1
            
            # Count by paper
            paper_counts[qa.paper_id] = paper_counts.get(qa.paper_id, 0) + 1
            
            # Length statistics
            total_question_length += len(qa.question.split())
            total_answer_length += len(qa.answer.split())
        
        return {
            "total_pairs": len(qa_pairs),
            "type_distribution": type_counts,
            "difficulty_distribution": difficulty_counts,
            "unique_papers": len(paper_counts),
            "avg_questions_per_paper": len(qa_pairs) / len(paper_counts),
            "avg_question_length": total_question_length / len(qa_pairs),
            "avg_answer_length": total_answer_length / len(qa_pairs),
            "total_question_words": total_question_length,
            "total_answer_words": total_answer_length
        }


def main():
    """Main function to create the dataset"""
    print("🤖 DocuMentor AI - Fine-tuning Dataset Creator")
    print("=" * 50)
    
    # Initialize creator
    creator = DatasetCreator()
    
    # Create dataset
    qa_pairs = creator.create_dataset(target_qa_pairs=1000)
    
    if qa_pairs:
        # Save in multiple formats
        saved_files = creator.save_dataset(qa_pairs)
        
        # Show statistics
        stats = creator.get_dataset_statistics(qa_pairs)
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total Q&A pairs: {stats['total_pairs']}")
        print(f"   Unique papers: {stats['unique_papers']}")
        print(f"   Avg questions per paper: {stats['avg_questions_per_paper']:.1f}")
        print(f"   Avg question length: {stats['avg_question_length']:.1f} words")
        print(f"   Avg answer length: {stats['avg_answer_length']:.1f} words")
        print(f"   Type distribution: {stats['type_distribution']}")
        print(f"   Difficulty distribution: {stats['difficulty_distribution']}")
        
        print(f"\n🎯 Fine-tuning dataset ready!")
        print(f"📁 Files saved: {len(saved_files)}")
        
    else:
        print("❌ No dataset created")


if __name__ == "__main__":
    main() 