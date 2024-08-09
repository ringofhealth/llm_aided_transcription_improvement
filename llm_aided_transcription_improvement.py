import os
import glob
import traceback
import asyncio
from aiolimiter import AsyncLimiter
import json
import re
import urllib.request
import logging
import warnings
from typing import List, Dict, Tuple, Optional
from llama_cpp import Llama, LlamaGrammar
import tiktoken
from decouple import Config as DecoupleConfig, RepositoryEnv
from filelock import FileLock, Timeout
from transformers import AutoTokenizer
from openai import AsyncOpenAI, APIError, RateLimitError
from anthropic import AsyncAnthropic
import backoff

try:
    import nvgpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Configuration
config = DecoupleConfig(RepositoryEnv('.env'))

USE_LOCAL_LLM = config.get("USE_LOCAL_LLM", default=False, cast=bool)
API_PROVIDER = config.get("API_PROVIDER", default="OPENAI", cast=str) # OPENAI or CLAUDE
ANTHROPIC_API_KEY = config.get("ANTHROPIC_API_KEY", default="your-anthropic-api-key", cast=str)
OPENAI_API_KEY = config.get("OPENAI_API_KEY", default="your-openai-api-key", cast=str)
CLAUDE_MODEL_STRING = config.get("CLAUDE_MODEL_STRING", default="claude-3-haiku-20240307", cast=str)
CLAUDE_MAX_TOKENS = 4096 # Maximum allowed tokens for Claude API
TOKEN_BUFFER = 500  # Buffer to account for token estimation inaccuracies
TOKEN_CUSHION = 300 # Don't use the full max tokens to avoid hitting the limit
OPENAI_COMPLETION_MODEL = config.get("OPENAI_COMPLETION_MODEL", default="gpt-4o-mini", cast=str)
OPENAI_EMBEDDING_MODEL = config.get("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small", cast=str)
OPENAI_MAX_TOKENS = 4096  # Maximum allowed tokens for OpenAI API
DEFAULT_LOCAL_MODEL_NAME = "Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS = 2048
USE_VERBOSE = False

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a rate limiter for API requests
rate_limit = AsyncLimiter(max_rate=60, time_period=60)  # 60 requests per minute

# GPU Check
def is_gpu_available():
    if not GPU_AVAILABLE:
        logging.warning("GPU support not available: nvgpu module not found")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": "nvgpu module not found"}
    try:
        gpu_info = nvgpu.gpu_info()
        num_gpus = len(gpu_info)
        if num_gpus == 0:
            logging.warning("No GPUs found on the system")
            return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0}
        first_gpu_vram = gpu_info[0]['mem_total']
        total_vram = sum(gpu['mem_total'] for gpu in gpu_info)
        logging.info(f"GPU(s) found: {num_gpus}, Total VRAM: {total_vram} MB")
        return {"gpu_found": True, "num_gpus": num_gpus, "first_gpu_vram": first_gpu_vram, "total_vram": total_vram, "gpu_info": gpu_info}
    except Exception as e:
        logging.error(f"Error checking GPU availability: {e}")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": str(e)}

# Model Download
async def download_models() -> Tuple[List[str], List[Dict[str, str]]]:
    download_status = []    
    model_url = "https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
    model_name = os.path.basename(model_url)
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(models_dir, exist_ok=True)
    lock = FileLock(os.path.join(models_dir, "download.lock"))
    status = {"url": model_url, "status": "success", "message": "File already exists."}
    filename = os.path.join(models_dir, model_name)
    
    try:
        with lock.acquire(timeout=1200):
            if not os.path.exists(filename):
                logging.info(f"Downloading model {model_name} from {model_url}...")
                urllib.request.urlretrieve(model_url, filename)
                file_size = os.path.getsize(filename) / (1024 * 1024)
                if file_size < 100:
                    os.remove(filename)
                    status["status"] = "failure"
                    status["message"] = f"Downloaded file is too small ({file_size:.2f} MB), probably not a valid model file."
                    logging.error(f"Error: {status['message']}")
                else:
                    logging.info(f"Successfully downloaded: {filename} (Size: {file_size:.2f} MB)")
            else:
                logging.info(f"Model file already exists: {filename}")
    except Timeout:
        logging.error(f"Error: Could not acquire lock for downloading {model_name}")
        status["status"] = "failure"
        status["message"] = "Could not acquire lock for downloading."
    
    download_status.append(status)
    logging.info("Model download process completed.")
    return [model_name], download_status

# Model Loading
def load_model(llm_model_name: str, raise_exception: bool = True):
    global USE_VERBOSE
    try:
        current_file_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(current_file_path)
        models_dir = os.path.join(base_dir, 'models')
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logging.error(f"Error: No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        model_file_path = max(matching_files, key=os.path.getmtime)
        logging.info(f"Loading model: {model_file_path}")
        try:
            logging.info("Attempting to load model with GPU acceleration...")
            model_instance = Llama(
                model_path=model_file_path,
                n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                verbose=USE_VERBOSE,
                n_gpu_layers=-1
            )
            logging.info("Model loaded successfully with GPU acceleration.")
        except Exception as gpu_e:
            logging.warning(f"Failed to load model with GPU acceleration: {gpu_e}")
            logging.info("Falling back to CPU...")
            try:
                model_instance = Llama(
                    model_path=model_file_path,
                    n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                    verbose=USE_VERBOSE,
                    n_gpu_layers=0
                )
                logging.info("Model loaded successfully with CPU.")
            except Exception as cpu_e:
                logging.error(f"Failed to load model with CPU: {cpu_e}")
                if raise_exception:
                    raise
                return None
        return model_instance
    except Exception as e:
        logging.error(f"Exception occurred while loading the model: {e}")
        traceback.print_exc()
        if raise_exception:
            raise
        return None

# API Interaction Functions
@backoff.on_exception(backoff.expo, 
                      (RateLimitError, APIError),
                      max_tries=5)
async def api_request_with_retry(client, *args, **kwargs):
    try:
        return await client(*args, **kwargs)
    except Exception as e:
        logging.error(f"API request failed after multiple retries: {str(e)}")
        raise

    
async def generate_completion(prompt: str, max_tokens: int = 5000) -> Optional[str]:
    if USE_LOCAL_LLM:
        return await generate_completion_from_local_llm(DEFAULT_LOCAL_MODEL_NAME, prompt, max_tokens)
    elif API_PROVIDER == "CLAUDE":
        safe_max_tokens = calculate_safe_max_tokens(len(prompt), CLAUDE_MAX_TOKENS)
        return await generate_completion_from_claude(prompt, safe_max_tokens)
    elif API_PROVIDER == "OPENAI":
        safe_max_tokens = calculate_safe_max_tokens(len(prompt), OPENAI_MAX_TOKENS)
        return await generate_completion_from_openai(prompt, safe_max_tokens)
    else:
        logging.error(f"Invalid API_PROVIDER: {API_PROVIDER}")
        return None

def get_tokenizer(model_name: str):
    if model_name.startswith("gpt-"):
        return tiktoken.encoding_for_model(model_name)
    elif model_name.startswith("claude-"):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=False)
    elif model_name.startswith("llama-"):
        return AutoTokenizer.from_pretrained("huggyllama/llama-7b", clean_up_tokenization_spaces=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def estimate_tokens(text: str, model_name: str) -> int:
    try:
        tokenizer = get_tokenizer(model_name)
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.warning(f"Error using tokenizer for {model_name}: {e}. Falling back to approximation.")
        return approximate_tokens(text)

def approximate_tokens(text: str) -> int:
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Split on whitespace and punctuation, keeping punctuation
    tokens = re.findall(r'\b\w+\b|\S', text)
    count = 0
    for token in tokens:
        if token.isdigit():
            count += max(1, len(token) // 2)  # Numbers often tokenize to multiple tokens
        elif re.match(r'^[A-Z]{2,}$', token):  # Acronyms
            count += len(token)
        elif re.search(r'[^\w\s]', token):  # Punctuation and special characters
            count += 1
        elif len(token) > 10:  # Long words often split into multiple tokens
            count += len(token) // 4 + 1
        else:
            count += 1
    # Add a 10% buffer for potential underestimation
    return int(count * 1.1)

def chunk_text(text: str, max_chunk_tokens: int, model_name: str) -> List[str]:
    chunks = []
    tokenizer = get_tokenizer(model_name)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_chunk_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if sentence_tokens > max_chunk_tokens:
            # If a single sentence is too long, split it into smaller parts
            sentence_parts = split_long_sentence(sentence, max_chunk_tokens, model_name)
            for part in sentence_parts:
                part_tokens = len(tokenizer.encode(part))
                if current_chunk_tokens + part_tokens > max_chunk_tokens:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [part]
                    current_chunk_tokens = part_tokens
                else:
                    current_chunk.append(part)
                    current_chunk_tokens += part_tokens
        elif current_chunk_tokens + sentence_tokens > max_chunk_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def split_long_sentence(sentence: str, max_tokens: int, model_name: str) -> List[str]:
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    tokenizer = get_tokenizer(model_name)
    
    for word in words:
        word_tokens = len(tokenizer.encode(word))
        if current_chunk_tokens + word_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_chunk_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_chunk_tokens += word_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def adjust_overlaps(chunks: List[str], tokenizer, max_chunk_tokens: int, overlap_size: int = 50) -> List[str]:
    adjusted_chunks = []
    for i in range(len(chunks)):
        if i == 0:
            adjusted_chunks.append(chunks[i])
        else:
            overlap_tokens = len(tokenizer.encode(' '.join(chunks[i-1].split()[-overlap_size:])))
            current_tokens = len(tokenizer.encode(chunks[i]))
            if overlap_tokens + current_tokens > max_chunk_tokens:
                overlap_adjusted = chunks[i].split()[:-overlap_size]
                adjusted_chunks.append(' '.join(overlap_adjusted))
            else:
                adjusted_chunks.append(' '.join(chunks[i-1].split()[-overlap_size:] + chunks[i].split()))
    
    return adjusted_chunks

async def generate_completion_from_claude(prompt: str, max_tokens: int = CLAUDE_MAX_TOKENS - TOKEN_BUFFER) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        logging.error("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        return None
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    prompt_tokens = estimate_tokens(prompt, CLAUDE_MODEL_STRING)
    adjusted_max_tokens = min(max_tokens, CLAUDE_MAX_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for Claude API. Chunking the input.")
        chunks = chunk_text(prompt, CLAUDE_MAX_TOKENS - TOKEN_CUSHION, CLAUDE_MODEL_STRING)
        results = []
        for chunk in chunks:
            try:
                async with client.messages.stream(
                    model=CLAUDE_MODEL_STRING,
                    max_tokens=CLAUDE_MAX_TOKENS // 2,
                    temperature=0.7,
                    messages=[{"role": "user", "content": chunk}],
                ) as stream:
                    message = await stream.get_final_message()
                    results.append(message.content[0].text)
                    logging.info(f"Chunk processed. Input tokens: {message.usage.input_tokens:,}, Output tokens: {message.usage.output_tokens:,}")
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        try:
            async with client.messages.stream(
                model=CLAUDE_MODEL_STRING,
                max_tokens=adjusted_max_tokens,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = await stream.get_final_message()
                output_text = message.content[0].text
                logging.info(f"Total input tokens: {message.usage.input_tokens:,}")
                logging.info(f"Total output tokens: {message.usage.output_tokens:,}")
                logging.info(f"Generated output (abbreviated): {output_text[:150]}...")
                return output_text
        except Exception as e:
            logging.error(f"An error occurred while requesting from Claude API: {e}")
            return None

async def generate_completion_from_openai(prompt: str, max_tokens: int = 5000) -> Optional[str]:
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    prompt_tokens = estimate_tokens(prompt, OPENAI_COMPLETION_MODEL)
    adjusted_max_tokens = min(max_tokens, 4096 - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for OpenAI API. Chunking the input.")
        chunks = chunk_text(prompt, OPENAI_MAX_TOKENS - TOKEN_CUSHION, OPENAI_COMPLETION_MODEL) 
        results = []
        for chunk in chunks:
            try:
                response = await api_request_with_retry(
                    openai_client.chat.completions.create,
                    model=OPENAI_COMPLETION_MODEL,
                    messages=[{"role": "user", "content": chunk}],
                    max_tokens=adjusted_max_tokens,
                    temperature=0.7,
                )
                result = response.choices[0].message.content
                results.append(result)
                logging.info(f"Chunk processed. Output tokens: {response.usage.completion_tokens:,}")
            except (RateLimitError, APIError) as e:
                logging.error(f"OpenAI API error: {str(e)}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing a chunk: {str(e)}")
        return " ".join(results)
    else:
        try:
            response = await api_request_with_retry(
                openai_client.chat.completions.create,
                model=OPENAI_COMPLETION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=adjusted_max_tokens,
                temperature=0.7,
            )
            output_text = response.choices[0].message.content
            logging.info(f"Total tokens: {response.usage.total_tokens:,}")
            logging.info(f"Generated output (abbreviated): {output_text[:150]}...")
            return output_text
        except (RateLimitError, APIError) as e:
            logging.error(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while requesting from OpenAI API: {str(e)}")
        return None

async def generate_completion_from_local_llm(llm_model_name: str, input_prompt: str, number_of_tokens_to_generate: int = 100, temperature: float = 0.7, grammar_file_string: str = None):
    logging.info(f"Starting text completion using model: '{llm_model_name}' for input prompt: '{input_prompt}'")
    llm = load_model(llm_model_name)
    prompt_tokens = estimate_tokens(input_prompt, llm_model_name)
    adjusted_max_tokens = min(number_of_tokens_to_generate, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for LLM. Chunking the input.")
        chunks = chunk_text(input_prompt, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION, llm_model_name)
        results = []
        for chunk in chunks:
            try:
                output = llm(
                    prompt=chunk,
                    max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION,
                    temperature=temperature,
                )
                results.append(output['choices'][0]['text'])
                logging.info(f"Chunk processed. Output tokens: {output['usage']['completion_tokens']:,}")
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        grammar_file_string_lower = grammar_file_string.lower() if grammar_file_string else ""
        if grammar_file_string_lower:
            list_of_grammar_files = glob.glob("./grammar_files/*.gbnf")
            matching_grammar_files = [x for x in list_of_grammar_files if grammar_file_string_lower in os.path.splitext(os.path.basename(x).lower())[0]]
            if len(matching_grammar_files) == 0:
                logging.error(f"No grammar file found matching: {grammar_file_string}")
                raise FileNotFoundError
            grammar_file_path = max(matching_grammar_files, key=os.path.getmtime)
            logging.info(f"Loading selected grammar file: '{grammar_file_path}'")
            llama_grammar = LlamaGrammar.from_file(grammar_file_path)
            output = llm(
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature,
                grammar=llama_grammar
            )
        else:
            output = llm(
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature
            )
        generated_text = output['choices'][0]['text']
        if grammar_file_string == 'json':
            generated_text = generated_text.encode('unicode_escape').decode()
        finish_reason = str(output['choices'][0]['finish_reason'])
        llm_model_usage_json = json.dumps(output['usage'])
        logging.info(f"Completed text completion in {output['usage']['total_time']:.2f} seconds. Beginning of generated text: \n'{generated_text[:150]}'...")
        return {
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "llm_model_usage_json": llm_model_usage_json
        }

def calculate_safe_max_tokens(input_length: int, model_max_tokens: int, token_buffer: int = 500) -> int:
    """Calculate a safe max_tokens value that won't exceed the model's limit."""
    available_tokens = max(0, model_max_tokens - input_length - token_buffer)
    safe_max = min(available_tokens, model_max_tokens // 2, 4096)  # Ensure we don't exceed OpenAI's max limit
    return max(1, safe_max)  # Ensure we always return at least 1 token

async def process_chunk_multi_stage(chunk: List[Dict], chunk_index: int, total_chunks: int) -> Tuple[int, str]:
    # Combine all text from the chunk
    original_text = " ".join(item["text"] for item in chunk)
    
    # Stage 1: Clean up
    stage1_prompt = f"""Clean up the following transcription chunk, improving readability. Follow these guidelines:
1. Combine utterances into coherent sentences and paragraphs with proper punctuation and capitalization (e.g., new line for paragraphs; new sentences start with capital letter and end with period or question mark or exclamation mark or ellipsis, etc.)
2. Add proper punctuation and capitalization
3. Remove filler words and obvious speech errors
4. Fix obvious transcription errors that were misheard or misunderstood
5. Improve the overall flow and coherence of the chunk

IMPORTANT: Use ONLY the information provided in the text below. Do not add any new content or make up any dialogue that is not present in the original text.

Original text:
{original_text}

Provide the cleaned up text, based strictly on the content above:
"""
    stage1_output = await generate_completion(stage1_prompt, max_tokens=calculate_safe_max_tokens(len(stage1_prompt), OPENAI_MAX_TOKENS))
    
    # Stage 2: Format as markdown
    stage2_prompt = f"""Format the following cleaned-up transcription as markdown. Follow these guidelines:
1. Use markdown headers for main topics or sections
2. Use markdown formatting for emphasis where appropriate
3. Use markdown list formatting for any lists in the content

IMPORTANT: Do not add any new content. Use only the information provided in the text below.

Cleaned up text:
{stage1_output}

Formatted markdown with proper formatting:
"""
    stage2_output = await generate_completion(stage2_prompt, max_tokens=calculate_safe_max_tokens(len(stage2_prompt), OPENAI_MAX_TOKENS))
    
    # Stage 3: Final refinement and consistency check
    stage3_prompt = f"""Refine the following markdown-formatted transcription chunk, ensuring consistency and improving readability. Follow these guidelines:
1. Check for and correct any formatting issues
2. Improve transitions between topics if necessary
3. Correct any obvious factual inconsistencies
4. Ensure the chunk flows well and is self-contained

IMPORTANT: Do not add any new content or alter the meaning of the text. Use only the information provided in the markdown below.

Markdown-formatted chunk:
{stage2_output}

Refined and consistent markdown chunk:
"""
    final_output = await generate_completion(stage3_prompt, max_tokens=calculate_safe_max_tokens(len(stage3_prompt), OPENAI_MAX_TOKENS))
    logging.info(f"Chunk {chunk_index + 1}/{total_chunks} processed through all stages")
    return chunk_index, final_output

async def process_chunks_parallel(chunks: List[List[Dict]]) -> List[str]:
    tasks = [process_chunk_multi_stage(chunk, i, len(chunks)) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    sorted_results = sorted(results, key=lambda x: x[0])
    return [result[1] for result in sorted_results]

def chunk_transcription(transcription: List[Dict], chunk_size: int = 10, overlap: int = 5) -> List[List[Dict]]:
    chunks = []
    for i in range(0, len(transcription), chunk_size - overlap):
        chunk = transcription[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

async def assess_output_quality(original_transcription: List[Dict], processed_text: str):
    max_chars = 10000  # Limit to avoid exceeding token limits
    original_sample = json.dumps(original_transcription[:5000], indent=2)[:max_chars // 2]
    processed_sample = processed_text[:max_chars // 2]
    
    prompt = f"""Compare the following samples of original transcription with the processed output and assess the quality of the processing. Consider the following factors:
1. Improvement in readability and coherence
2. Appropriate use of markdown formatting
3. Preservation of original content and meaning
4. Removal of filler words and speech errors
5. Overall structure and flow of the document

Original transcription sample:
```json
{original_sample}
```

Processed text sample:
```markdown
{processed_sample}
```

Provide a quality score between 0 and 100, where 100 is perfect processing. Also provide a brief explanation of your assessment.

Your response should be in the following format:
SCORE: [Your score]
EXPLANATION: [Your explanation]
"""

    response = await generate_completion(prompt, max_tokens=1000)
    
    print("Raw response from the model:")
    print(response)
    logging.info(f"Raw response from the model:\n{response}")
    
    return response  # Return the raw response for further analysis if needed

async def process_long_text(text: str, max_chunk_size: int = 2000) -> str:
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    processed_chunks = []
    
    for i, chunk in enumerate(chunks):
        logging.info(f"Processing long text chunk {i+1}/{len(chunks)}")
        prompt = f"""Refine and improve the following chunk of text, ensuring consistency in formatting and content:

{chunk}

Refined chunk:"""
        processed_chunk = await generate_completion(prompt, max_tokens=max_chunk_size)
        if processed_chunk:
            processed_chunks.append(processed_chunk)
        else:
            logging.error(f"Failed to process chunk {i+1}/{len(chunks)}")
            processed_chunks.append(chunk)  # Use original chunk if processing fails
    
    return "\n\n".join(processed_chunks)

async def main():
    # Suppress HTTP request logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Load the transcription
    with open('example_whisper_transcription.json', 'r') as f:
        transcription = json.load(f)
    
    # Split into chunks
    chunks = chunk_transcription(transcription)
    
    # Process chunks in parallel
    processed_chunks = await process_chunks_parallel(chunks)
    
    # Combine processed chunks
    final_text = "\n\n".join(processed_chunks)
    
    # Write to output file
    with open('formatted_transcription.md', 'w') as f:
        f.write(final_text)
    
    logging.info("Processing complete. Output written to formatted_transcription.md")
    
    # Optionally, you can still run a quality assessment on the final output
    quality_assessment = await assess_output_quality(transcription, final_text)
    logging.info(f"Quality assessment results: {quality_assessment}")

if __name__ == '__main__':
    asyncio.run(main())
if __name__ == '__main__':
    asyncio.run(main())