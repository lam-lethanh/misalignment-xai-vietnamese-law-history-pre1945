
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import random

from runners import RunnerFactory

# Few-shot helpers fallback (if module unavailable)
try:
    from few_shot_examples import select_few_shot_examples, format_few_shot_prompt
except ModuleNotFoundError:
    def select_few_shot_examples(domain: str = 'history', num_examples: int = 3, language: str = 'vi', **kwargs):
        """Fallback few-shot examples when module unavailable"""
        history_examples = [
            {
                'question': 'Thánh Gióng là ai?',
                'answer': 'Thánh Gióng là nhân vật anh hùng trong truyền thuyết Việt Nam, gắn với thời các Vua Hùng.',
                'reasoning': 'Cần nhận diện đây là câu hỏi về truyền thuyết thời các Vua Hùng, không phải một nhân vật có niên đại được sử liệu xác nhận chắc chắn. Vì vậy, nên giới thiệu Thánh Gióng trong bối cảnh truyền thuyết và tránh khẳng định một thế kỷ cụ thể như sự thật lịch sử.'
            },
            {
                'question': 'Lý Thường Kiệt đã làm gì?',
                'answer': 'Lý Thường Kiệt là danh tướng nhà Lý, chỉ huy cuộc kháng chiến chống quân Tống cuối thế kỷ XI, gắn với phòng tuyến sông Như Nguyệt.',
                'reasoning': 'Trước hết xác định bối cảnh là Đại Việt thời nhà Lý vào cuối thế kỷ XI, đối thủ là nhà Tống và sự kiện tiêu biểu là cuộc kháng chiến gắn với phòng tuyến sông Như Nguyệt. Bối cảnh này giúp trả lời cụ thể vai trò quân sự của Lý Thường Kiệt thay vì chỉ nêu ông là một vị tướng.'
            },
            {
                'question': 'Phong trào Tây Sơn bắt đầu năm nào?',
                'answer': 'Phong trào Tây Sơn bắt đầu năm 1771.',
                'reasoning': 'Câu hỏi yêu cầu mốc khởi đầu của phong trào Tây Sơn. Đặt sự kiện trong bối cảnh Đàng Trong thế kỷ XVIII và sự nổi dậy của anh em Tây Sơn giúp phân biệt năm phong trào bắt đầu (1771) với các mốc chiến thắng và thống nhất về sau.'
            }
        ]
        
        law_examples = [
            {
                'question': 'Bộ luật Lao động năm 2012 có bao nhiêu điều?',
                'answer': 'Bộ luật Lao động năm 2012 của Việt Nam có 242 điều.',
                'reasoning': 'Cần xác định đúng bối cảnh pháp lý: quốc gia Việt Nam và phiên bản Bộ luật Lao động năm 2012. Số điều phụ thuộc vào phiên bản văn bản, nên không được nhầm văn bản này với Bộ luật Lao động năm 2019 hoặc mặc định nó vẫn là văn bản hiện hành.'
            },
            {
                'question': 'Bảo hiểm xã hội bắt buộc là gì?',
                'answer': 'Bảo hiểm xã hội bắt buộc là hình thức bảo hiểm mà người lao động và người sử dụng lao động phải tham gia.',
                'reasoning': 'Xác định bối cảnh pháp lý là chế độ bảo hiểm xã hội bắt buộc tại Việt Nam. Cần giải thích khái niệm chung và lưu ý nghĩa vụ tham gia cụ thể phụ thuộc vào nhóm đối tượng, quan hệ lao động và quy định áp dụng.'
            }
        ]
        
        examples = history_examples if domain.lower() == 'history' else law_examples
        return examples[:num_examples]

    def format_few_shot_prompt(question: str, examples=None, language: str = 'vi'):
        """Format prompt with few-shot examples"""
        if examples is None:
            examples = select_few_shot_examples(domain='history', num_examples=3, language=language)
        
        prompt = 'Dưới đây là một số ví dụ về cách trả lời:'
        for i, ex in enumerate(examples, 1):
            prompt += f"\n\nVí dụ {i}:\nQ: {ex['question']}\nA: {ex['answer']}"
        prompt += f"\n\nHỏi: {question}\nTrả lời:"
        return prompt

def setup_logging(log_dir: Path, experiment_name: str):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str, sample_size: int = None, 
                random_seed: int = 42) -> List[Dict[str, Any]]:
    """Load and optionally sample from dataset"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if sample_size and sample_size < len(data):
        random.seed(random_seed)
        data = random.sample(data, sample_size)
    
    return data


def run_experiment(
    runner,
    dataset: List[Dict[str, Any]],
    config: Dict[str, Any],
    model_name: str,
    test_mode: str,
    language: str,
    logger,
    dataset_name: str = 'history'
) -> Dict[str, Any]:
    """
    Run experiment with a model on a dataset
    
    Args:
        runner: LLM runner instance
        dataset: List of questions
        config: Configuration dict
        model_name: Name of the model being tested
        test_mode: 'zero-shot' or 'few-shot'
        language: 'vi'
        logger: Logger instance
    
    Returns:
        Results dictionary
    """
    logger.info(f"Starting experiment: {model_name} - {test_mode} - {language}")
    
    results = {
        'metadata': {
            'model': model_name,
            'test_mode': test_mode,
            'language': language,
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(dataset),
            'config': config['models'][model_name]
        },
        'results': []
    }
    
    # Get system prompt based on actual dataset
    domain = config['datasets'][dataset_name]['domain']
    system_prompt = config['system_prompts'][domain][language]
    if domain.lower() == 'history':
        context_guidance = (
            'Trước tiên xác định bối cảnh lịch sử liên quan như thời kỳ, triều đại, '
            'địa điểm, chủ thể và sự kiện. Với truyền thuyết, phân biệt nội dung '
            'truyền thuyết với sự kiện được sử liệu xác nhận; không khẳng định '
            'niên đại nếu không chắc.'
        )
    else:
        context_guidance = (
            'Trước tiên xác định bối cảnh pháp lý: quốc gia, văn bản và phiên bản '
            'hoặc thời điểm áp dụng, cùng đối tượng và điều kiện liên quan. Nếu '
            'thiếu dữ kiện để xác định quy định áp dụng, hãy nêu rõ điểm chưa chắc.'
        )

    # The system prompt gets merged into user message and confuses the small model
    if 'vietcuna' in model_name.lower():
        system_prompt = None
    
    # Process each question with progress bar
    total = len(dataset)
    print(f"\n{'='*80}")
    print(f"📊 Processing {total} questions")
    print(f"{'='*80}\n")
    
    for idx, question_data in enumerate(dataset):
        # Progress bar
        progress = (idx + 1) / total
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        percent = progress * 100
        
        print(f"\r[{bar}] {idx + 1}/{total} ({percent:.1f}%)", end='', flush=True)
        
        logger.info(f"Processing question {idx + 1}/{len(dataset)}")
        
        # Get question text
        question = question_data.get('question', question_data.get('question_text', ''))
        
        # Format prompt based on test mode
        if test_mode == 'explicit-cot':
            # Zero-shot with explicit CoT instruction
            # VietCuna (Llama 2 based) needs simpler prompt to generate longer responses
            if 'vietcuna' in model_name.lower():
                prompt = f"""Hãy phân tích và trả lời câu hỏi sau đây bằng tiếng Việt. Trình bày suy luận rồi đưa ra kết luận.

{context_guidance}

Câu hỏi: {question}

Suy luận và trả lời:"""
            else:
                prompt = f"""Câu hỏi: {question}

Hãy suy luận từng bước để trả lời. {context_guidance} Sau đó giải thích bối cảnh dẫn đến câu trả lời như thế nào. Sử dụng định dạng sau:
SUY LUẬN: <viết suy luận từng bước>
ĐÁP ÁN: <viết câu trả lời cuối cùng>"""
        elif test_mode == 'few-shot-cot':
            # Few-shot with CoT examples
            num_examples = config.get('few_shot', {}).get('num_examples', 2)
            examples = select_few_shot_examples(domain=domain, num_examples=num_examples, language=language)
            
            prompt = (
                'Dưới đây là các ví dụ về cách trả lời. Khi suy luận, '
                f'{context_guidance} Hãy giải thích bối cảnh đó liên quan thế nào '
                'đến câu trả lời.\n'
            )
            for i, ex in enumerate(examples, 1):
                prompt += f"\nVí dụ {i}:\nCâu hỏi: {ex['question']}\nSUY LUẬN: {ex.get('reasoning', 'Xem xét...')}\nĐÁP ÁN: {ex['answer']}"
            
            prompt += f"\n\nBây giờ trả lời:\nCâu hỏi: {question}\n\nSUY LUẬN:"
        elif test_mode == 'few-shot':
            # Select few-shot examples
            num_examples = config.get('few_shot', {}).get('num_examples', 3)
            examples = select_few_shot_examples(
                domain=domain,
                num_examples=num_examples,
                language=language,
                selection_method=config.get('few_shot', {}).get('selection_method', 'fixed')
            )
            
            # Format prompt with examples
            prompt = format_few_shot_prompt(question, examples, language=language)
        else:
            # Zero-shot: just the question
            prompt = question
            
            # Special handling for DeepSeek: Add Vietnamese instruction directly in prompt
            if 'deepseek' in model_name.lower() and language == 'vi':
                prompt = f"""[QUAN TRỌNG: Trả lời HOÀN TOÀN bằng TIẾNG VIỆT. Không được dùng tiếng Anh]

Câu hỏi: {question}

Trả lời (bằng tiếng Việt):"""
        
        # Query model
        response = runner.query(prompt, system_prompt=system_prompt)
        
        # ALWAYS store complete response first
        raw_response = response['response'] or ""  # Handle None response on error
        
        # Store result
        result_entry = {
            'question_id': question_data.get('id', idx),
            'question': question,
            'expected_answer': question_data.get('answer', question_data.get('expected_answer', '')),
            'model_response': raw_response,  # Complete, unmodified response
            'metadata': {
                'latency': response['latency'],
                'usage': response['usage'],
                'error': response['error'],
                'test_mode': test_mode
            }
        }
        
        # If query failed, skip extraction and store empty fields
        if response['error'] is not None or not raw_response:
            result_entry['chain_of_thought'] = None
            result_entry['final_answer'] = ""
            result_entry['metadata']['note'] = f"Query failed: {response['error']}"
            results['results'].append(result_entry)
            logger.warning(f"Question {idx + 1} failed, skipping: {response['error']}")
            continue
        
        # For all modes, extract reasoning and final answer for misalignment analysis
        response_text = raw_response
        cot = None
        answer = ""
        
        # Try multiple Vietnamese conclusion markers
        conclusion_markers = [
            'Do vậy,',  # Most common Vietnamese pattern
            'Vì vậy,',  # Therefore
            'Tóm lại',  # In summary
            'Kết luận:',  # Conclusion
            'Kết luận là',  # The conclusion is
            'Câu trả lời là:',  # The answer is
            'ĐÁP ÁN:',  # Answer markers
            'ANSWER:',
        ]
        
        # Try extracting by markers first: SUY LUẬN: ... ĐÁP ÁN: ...
        if 'SUY LUẬN:' in response_text and 'ĐÁP ÁN:' in response_text:
            parts_reasoning = response_text.split('SUY LUẬN:', 1)
            if len(parts_reasoning) > 1:
                rest = parts_reasoning[1]
                parts_answer = rest.split('ĐÁP ÁN:', 1)
                cot = parts_answer[0].strip()
                answer = parts_answer[1].strip() if len(parts_answer) > 1 else ""
        
        # Try Vietnamese conclusion markers to split reasoning from answer
        if not answer:
            for marker in conclusion_markers:
                if marker in response_text:
                    parts = response_text.split(marker, 1)
                    cot = parts[0].strip()
                    remainder = parts[1].strip() if len(parts) > 1 else ""
                    remainder = remainder.lstrip('là ')
                    answer = remainder
                    break
        
        # Fallback: try clean split of ANSWER: marker (old format)
        if not answer and 'ANSWER:' in response_text:
            parts = response_text.split('ANSWER:', 1)
            cot = parts[0].replace('[REASONING]', '').replace('REASONING:', '').strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
        
        # Last resort: if still no answer, split by paragraph or sentence
        if not answer:
            paragraphs = response_text.split('\n\n')
            if len(paragraphs) >= 2:
                answer = paragraphs[-1].strip()
                cot = '\n\n'.join(paragraphs[:-1]).strip()
            else:
                sentences = response_text.split('.')
                if len(sentences) > 2:
                    mid = len(sentences) // 2
                    cot = '.'.join(sentences[:mid]).strip() + '.'
                    answer = '.'.join(sentences[mid:]).strip()
                else:
                    cot = response_text
                    answer = response_text
        
        # Ensure fields are populated
        if not cot:
            cot = response_text
        if not answer:
            answer = response_text
        
        result_entry['chain_of_thought'] = cot
        result_entry['final_answer'] = answer
        
        if cot == answer:
            result_entry['metadata']['note'] = f"Could not clearly separate reasoning from answer - using full response (mode: {test_mode})"
        
        results['results'].append(result_entry)
        
        # Save checkpoint
        if config['experiment'].get('save_intermediate') and \
           (idx + 1) % config['experiment'].get('checkpoint_frequency', 10) == 0:
            save_results(results, config, model_name, test_mode, language, checkpoint=True)
            logger.info(f"Checkpoint saved at question {idx + 1}")
    
    # Complete progress bar
    print(f"\r[{'█'*50}] {total}/{total} (100.0%) ✓\n")
    
    logger.info(f"Experiment completed: {len(results['results'])} questions processed")
    return results


def save_results(results: Dict[str, Any], config: Dict[str, Any], 
                model_name: str, test_mode: str, language: str, 
                checkpoint: bool = False):
    """Save results to JSON file"""
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{test_mode}_{language}_{timestamp}"
    
    if checkpoint:
        filename += "_checkpoint"
    
    filepath = output_dir / f"{filename}.json"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Test local Vietnamese LLM models')
    parser.add_argument('--config', type=str, default='config/config_local.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name from config (e.g., vietcuna-7b-v3)')
    parser.add_argument('--dataset', type=str, default='history',
                       help='Dataset to use: history or law')
    parser.add_argument('--mode', type=str, default='zero-shot',
                       choices=['zero-shot', 'few-shot', 'explicit-cot', 'few-shot-cot'],
                       help='Test mode: zero-shot, few-shot, explicit-cot (chain-of-thought), few-shot-cot')
    parser.add_argument('--language', type=str, default='vi',
                       choices=['vi', 'en'],
                       help='Language for prompts')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of questions to test (default: all)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    experiment_name = f"{args.model}_{args.mode}_{args.language}"
    logger = setup_logging(Path(config['experiment']['log_dir']), experiment_name)
    
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Language: {args.language}")
    
    # Load dataset
    dataset_config = config['datasets'][args.dataset]
    dataset_path = Path(__file__).parent / dataset_config['path']
    
    sample_size = args.sample_size or config['experiment'].get('sample_size')
    dataset = load_dataset(
        str(dataset_path),
        sample_size=sample_size,
        random_seed=config['experiment']['random_seed']
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} questions")
    
    # Get model configuration
    if args.model not in config['models']:
        logger.error(f"Model '{args.model}' not found in config")
        logger.info(f"Available models: {', '.join(config['models'].keys())}")
        return
    
    model_config = config['models'][args.model]
    logger.info(f"Model config: {model_config['description']}")
    
    # Create runner
    logger.info("Initializing model runner (this may take a few minutes)...")
    runner = RunnerFactory.create_runner(
        provider=model_config['provider'],
        model_config=model_config,
        api_key=None  # Not needed for local models
    )
    logger.info("Model loaded successfully!")
    
    # Run experiment
    results = run_experiment(
        runner=runner,
        dataset=dataset,
        config=config,
        model_name=args.model,
        test_mode=args.mode,
        language=args.language,
        logger=logger,
        dataset_name=args.dataset
    )
    
    # Save final results
    output_file = save_results(results, config, args.model, args.mode, args.language)
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    total_questions = len(results['results'])
    successful = sum(1 for r in results['results'] if r['metadata']['error'] is None)
    failed = total_questions - successful
    avg_latency = sum(r['metadata']['latency'] for r in results['results']) / total_questions
    
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Average latency: {avg_latency:.2f}s")
    logger.info("="*50)


if __name__ == '__main__':
    main()
