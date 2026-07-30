
from typing import Dict, Any, Optional
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base_runner import BaseLLMRunner
import logging

class LocalModelRunner(BaseLLMRunner):    
    def __init__(self, model_config: Dict[str, Any], api_key: str = None):
        super().__init__(model_config, api_key)
        
        # Cấu hình quantization để giảm VRAM
        self.use_quantization = model_config.get('use_quantization', True)
        self.load_in_8bit = model_config.get('load_in_8bit', False)
        self.load_in_4bit = model_config.get('load_in_4bit', True)
        self.device = model_config.get('device', 'auto')
        
        # Cấu hình precision (torch dtype)
        dtype_str = model_config.get('torch_dtype', 'float16')
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32
        }
        self.torch_dtype = dtype_map.get(dtype_str, torch.float16)
        
        # Cấu hình generation
        self.do_sample = model_config.get('do_sample', True)
        self.top_p = model_config.get('top_p', 0.9)
        self.top_k = model_config.get('top_k', 50)
        self.repetition_penalty = model_config.get('repetition_penalty', 1.1)
        self.min_p = model_config.get('min_p', 0.0)
        self.epsilon_cutoff = model_config.get('epsilon_cutoff', 0.0)
        self.use_cache_gen = model_config.get('use_cache', True)
        self.trust_remote_code = model_config.get('trust_remote_code', True)
        self.attn_implementation = model_config.get('attn_implementation', None)
        
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load model và tokenizer từ Hugging Face"""
        self.logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Check local path support
            local_files_only = False
            model_path = Path(self.model_name)
            if model_path.exists():
                self.logger.info(f"Detected local model path: {model_path}")
                self.model_name = str(model_path.resolve())
                local_files_only = True

            # Load tokenizer
            tokenizer_kwargs = {
                'trust_remote_code': self.trust_remote_code,
            }
            if local_files_only:
                tokenizer_kwargs['local_files_only'] = True

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs
            )
            
            # Đảm bảo có padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set default chat template for models that don't have one
            if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
                self.logger.info("Set default chat template for model")
            
            # Cấu hình quantization nếu được bật
            quantization_config = None
            if self.use_quantization:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=self.load_in_8bit,
                        load_in_4bit=self.load_in_4bit,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                except Exception as e:
                    self.logger.warning(f"Quantization not supported on this platform: {str(e)}")
                    self.logger.warning("Loading model in float16 instead")
                    quantization_config = None
            
            # Load model
            model_kwargs = {
                'quantization_config': quantization_config,
                'device_map': self.device if quantization_config else None,
                'trust_remote_code': self.trust_remote_code,
                'torch_dtype': self.torch_dtype  # Use configured dtype instead of hardcoded float16
            }
            
            # Add attn_implementation if specified (for Phi-3 compatibility)
            if self.attn_implementation:
                model_kwargs['attn_implementation'] = self.attn_implementation
            
            if local_files_only:
                model_kwargs['local_files_only'] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device nếu không dùng quantization
            if quantization_config is None and self.device != 'auto':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.logger.info(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Query model locally với retry logic
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Response dictionary với response, usage, latency, và error fields
        """
        result = {
            'response': None,
            'usage': None,
            'latency': None,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            # Format messages theo template của model
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Apply chat template nếu có
            if hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    # Try với system role trước
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    # Nếu lỗi "System role not supported", retry without system
                    if "system" in str(e).lower() or "role" in str(e).lower():
                        self.logger.warning(f"Model không hỗ trợ system role, retry without system prompt")
                        # Prepend system prompt vào user message
                        user_msg = messages[-1]["content"]
                        if system_prompt:
                            user_msg = f"{system_prompt}\n\n{user_msg}"
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": user_msg}],
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        raise  # Re-raise nếu lỗi khác
            else:
                # Fallback format nếu không có chat template
                if system_prompt:
                    formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                else:
                    formatted_prompt = f"User: {prompt}\n\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate (with retry on inf/nan errors)
            with torch.no_grad():
                generation_config = {
                    'max_new_tokens': self.max_tokens,
                    'temperature': self.temperature,
                    'do_sample': self.do_sample,
                    'top_p': self.top_p,
                    'top_k': self.top_k,
                    'repetition_penalty': self.repetition_penalty,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'use_cache': self.use_cache_gen
                }
                
                # Add optional parameters if specified
                if isinstance(self.min_p, (int, float)) and self.min_p > 0:
                    generation_config['min_p'] = float(self.min_p)
                if isinstance(self.epsilon_cutoff, (int, float)) and self.epsilon_cutoff > 0:
                    generation_config['epsilon_cutoff'] = float(self.epsilon_cutoff)
                
                try:
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                except RuntimeError as gen_err:
                    if 'inf' in str(gen_err) or 'nan' in str(gen_err) or 'probability tensor' in str(gen_err):
                        # Retry with greedy decoding (no sampling) to avoid nan/inf
                        self.logger.warning(f"inf/nan error, retrying with greedy decoding: {gen_err}")
                        safe_config = {
                            'max_new_tokens': self.max_tokens,
                            'do_sample': False,  # Greedy decoding - no sampling
                            'pad_token_id': self.tokenizer.pad_token_id,
                            'eos_token_id': self.tokenizer.eos_token_id,
                        }
                        outputs = self.model.generate(**inputs, **safe_config)
                    else:
                        raise
            
            # Decode response - only decode the generated tokens to avoid truncation
            input_length = inputs['input_ids'].shape[1]
            output_token_ids = outputs[0, input_length:]  # Only the newly generated tokens
            response = self.tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()
            
            result['response'] = response
            
            # Token usage
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = outputs.shape[1] - input_tokens
            result['usage'] = {
                'prompt_tokens': input_tokens,
                'completion_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            }
            
            result['latency'] = time.time() - start_time
            self.logger.info(f"Query successful - Latency: {result['latency']:.2f}s, "
                           f"Tokens: {result['usage']['total_tokens']}")
            
        except Exception as e:
            result['error'] = str(e)
            result['latency'] = time.time() - start_time
            self.logger.error(f"Query failed: {str(e)}")
        
        return result
    
    def __del__(self):
        """Cleanup khi object bị destroy"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
