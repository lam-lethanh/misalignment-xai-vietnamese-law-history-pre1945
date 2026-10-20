#!/usr/bin/env python3
"""
Batch test multiple local models and compare results
Chạy test với nhiều models và so sánh kết quả
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

# Models to test - TRUE open source (NO gated repos)
MODELS = [
    # === Standard 7B Models ===
    {
        "name": "qwen-2.5-7b-instruct",
        "description": "Qwen 2.5 7B (multilingual SOTA)",
        "size": "7B"
    },
    {
        "name": "vietcuna-7b-v3",
        "description": "VietCuna 7B (Vietnamese fine-tuned)",
        "size": "7B"
    },
    {
        "name": "llama-2-7b-chat-hf",
        "description": "Llama 2 7B (Meta)",
        "size": "7B"
    },
    {
        "name": "gemma2-9b-instruct",
        "description": "Gemma 2 9B (Google)",
        "size": "9B"
    }
]

def run_test(model_name: str, dataset: str = "history", mode: str = "zero-shot", sample_size: int = None):
    """Run test for a single model on a dataset with specific mode"""
    print(f"\n{'='*80}")
    print(f"🚀 Testing: {model_name} on {dataset} dataset ({mode})")
    if sample_size:
        print(f"📊 Sample size: {sample_size}")
    else:
        print(f"📊 Full dataset (250 questions)")
    print(f"{'='*80}")
    
    cmd = [
        "python3", "run_local_test.py",
        "--model", model_name,
        "--dataset", dataset,
        "--mode", mode
    ]
    
    if sample_size:
        cmd.extend(["--sample-size", str(sample_size)])
    
    # No timeout - let it run to completion
    # Real-time output for progress tracking
    print(f"\n⏳ Running... (this may take a while)\n")
    
    try:
        result = subprocess.run(
            cmd,
            text=True,
            # Let output show in real-time
            stdout=None,
            stderr=None
        )
        
        if result.returncode == 0:
            print(f"\n✅ Success: {model_name} on {dataset} ({mode})")
            return True, "Success"
        else:
            print(f"\n❌ Failed: {model_name} on {dataset} ({mode})")
            return False, "Failed"
            
    except Exception as e:
        print(f"\n❌ Error: {model_name} - {str(e)}")
        return False, str(e)

def main():
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "BATCH MODEL TESTING WITH CoT (Chain of Thought)" + " "*16 + "║")
    print("╚" + "="*78 + "╝")
    
    # Get arguments: sample_size (or 'full'), datasets, modes
    sample_size = None  # None = full dataset
    datasets = ["history", "law"]
    modes = ["zero-shot", "explicit-cot", "few-shot-cot"]  # CoT modes
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() != "full":
            try:
                sample_size = int(arg)
            except:
                pass
    
    if len(sys.argv) > 2:
        datasets = sys.argv[2].split(",")
    
    if len(sys.argv) > 3:
        modes = sys.argv[3].split(",")
    
    dataset_info = f"full dataset (250 questions)" if sample_size is None else f"{sample_size} samples"
    print(f"\n📊 Testing {len(MODELS)} models x {len(datasets)} datasets x {len(modes)} modes")
    print(f"📈 Dataset size: {dataset_info}")
    print(f"📁 Datasets: {', '.join(datasets)}")
    print(f"🧠 Modes: {', '.join(modes)}")
    print(f"⏱️  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = []
    
    for dataset in datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*80}")
        
        for mode in modes:
            print(f"\n{'*'*80}")
            print(f"* MODE: {mode.upper()}")
            print(f"{'*'*80}")
            
            for i, model_info in enumerate(MODELS, 1):
                print(f"\n[{i}/{len(MODELS)}] Model: {model_info['name']}")
                print(f"    Description: {model_info['description']}")
                print(f"    Size: {model_info['size']}")
                
                success, output = run_test(model_info['name'], dataset, mode, sample_size)
                
                results.append({
                    "model": model_info['name'],
                    "dataset": dataset,
                    "mode": mode,
                    "description": model_info['description'],
                    "size": model_info['size'],
                    "success": success,
                    "output": output[:1000] if output else ""
                })
                
                if not success:
                    print(f"⚠️  Skipping to next mode due to error")
                    break
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    for r in successful[:10]:  # Show first 10
        print(f"   • {r['model']:25} | {r['dataset']:7} | {r['mode']:15} ({r['size']})")
    if len(successful) > 10:
        print(f"   ... and {len(successful) - 10} more")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for r in failed[:10]:  # Show first 10
            print(f"   • {r['model']:25} | {r['dataset']:7} | {r['mode']:15} ({r['size']})")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")
    
    print(f"\n⏱️  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results summary
    summary_file = Path("results") / f"batch_test_cot_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_file.parent.mkdir(exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'sample_size': sample_size if sample_size else "full (250 questions)",
            'datasets': datasets,
            'modes': modes,
            'total_results': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Statistics by mode
    print("\n" + "="*80)
    print("📈 SUCCESS RATE BY MODE")
    print("="*80)
    for mode in modes:
        mode_results = [r for r in results if r['mode'] == mode]
        mode_success = [r for r in mode_results if r['success']]
        rate = (len(mode_success) / len(mode_results) * 100) if mode_results else 0
        print(f"  {mode:20} : {len(mode_success):2}/{len(mode_results):2} ({rate:5.1f}%)")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
