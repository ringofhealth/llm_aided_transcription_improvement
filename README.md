# LLM-Aided Transcription Improvement

## Introduction

The LLM-Aided Transcription Improvement Project is an advanced system designed to significantly enhance the quality of raw audio transcription output, like that generated by OpenAI's Whisper model. By applying a pipeline of LLM prompts, the system corrects errors, improves readability, and formats the text in markdown for easy consumption. The project supports both local LLMs and cloud-based API providers, allowing for flexible integration with various models, but OpenAI's GPT-4o-mini is the default choice and works very well and at low cost. The system is designed to handle large transcription files efficiently, with parallel processing and adaptive token management for optimal performance. It's particularly useful for processing the transcripts generated by my other project, [bulk_transcribe_youtube_videos_from_playlist](https://github.com/Dicklesworthstone/bulk_transcribe_youtube_videos_from_playlist), which takes as input either a single YouTube video or a playlist of videos and generates a transcription file for each video in the playlist. Because the transcription files are in JSON format, they can be easily processed by this project, which is able to break them into chunks and process them in parallel, making the process much faster and more efficient.

## Example Outputs

To see what the LLM-Aided Transcription Project can do, check out these example outputs:

- [Original Transcript JSON File (Output from Whisper)](https://github.com/Dicklesworthstone/llm_aided_transcription_improvement/blob/main/example_whisper_transcription.json) 
- [Final LLM Generated Markdown Transcript](https://github.com/Dicklesworthstone/llm_aided_transcription_improvement/blob/main/formatted_transcription.md)

## Features

- Multi-stage processing of transcription chunks
- Advanced error correction and formatting using LLMs
- Support for both local LLMs and cloud-based API providers (OpenAI, Anthropic)
- Smart text chunking for efficient processing
- Markdown formatting
- Asynchronous processing for improved performance
- Detailed logging for process tracking and debugging
- Quality assessment of the final output

## Detailed Technical Overview

### Transcription Processing Pipeline

1. **Chunk Creation**
   - Function: `chunk_transcription()`
   - Splits the full transcription into manageable chunks
   - Implements an overlap between chunks to maintain context

2. **Multi-Stage Processing**
   - Core function: `process_chunk_multi_stage()`
   - Three-step process:
     a. Clean-up:
        - Improves readability and coherence
        - Fixes obvious transcription errors
     b. Markdown Formatting:
        - Converts text to proper markdown format
        - Handles headings, lists, emphasis, and more
     c. Final Refinement:
        - Ensures consistency across the chunk
        - Improves transitions and flow

3. **Parallel Processing**
   - Function: `process_chunks_parallel()`
   - Uses `asyncio` for concurrent processing of chunks
   - Maintains order of processed chunks for coherent final output

### LLM Integration

1. **Flexible LLM Support**
   - Supports both local LLMs and cloud-based API providers (OpenAI, Anthropic)
   - Configurable through environment variables

2. **Local LLM Handling**
   - Function: `generate_completion_from_local_llm()`
   - Uses `llama_cpp` library for local LLM inference
   - Supports custom grammars for structured output

3. **API-based LLM Handling**
   - Functions: `generate_completion_from_claude()` and `generate_completion_from_openai()`
   - Implements proper error handling and retry logic
   - Manages token limits and adjusts request sizes dynamically

### Token Management

1. **Token Estimation**
   - Function: `estimate_tokens()`
   - Uses model-specific tokenizers when available
   - Falls back to `approximate_tokens()` for quick estimation

2. **Dynamic Token Adjustment**
   - Function: `calculate_safe_max_tokens()`
   - Adjusts `max_tokens` parameter based on prompt length and model limits
   - Implements `TOKEN_BUFFER` for safe token management

### Quality Assessment

1. **Output Quality Evaluation**
   - Function: `assess_output_quality()`
   - Compares original transcription text with processed output
   - Uses LLM to provide a quality score and explanation

### Logging and Error Handling

- Comprehensive logging throughout the codebase
- Detailed error messages and stack traces for debugging
- Suppresses HTTP request logs to reduce noise

## Configuration and Customization

The project uses a `.env` file for easy configuration. Key settings include:

- LLM selection (local or API-based)
- API provider selection (OpenAI or Anthropic)
- Model selection for different providers
- Token limits and buffer sizes

## Output and File Handling

1. **Input**: JSON file containing the raw transcription (`example_whisper_transcription.json`)
2. **Output**: Formatted transcription saved as `formatted_transcription.md`

The script generates detailed logs of the entire process, including timing information and quality assessments.

## Requirements

- Python 3.12+
- OpenAI API (optional)
- Anthropic API (optional)
- Local LLM support (optional, requires compatible GGUF model)
- Various Python libraries (see `requirements.txt`)

## Installation

1. Install Pyenv and Python 3.12 (if needed):

```bash
# Install Pyenv and python 3.12 if needed and then use it to create venv:
if ! command -v pyenv &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
    source ~/.zshrc
fi
cd ~/.pyenv && git pull && cd -
pyenv install 3.12
```

2. Set up the project:

```bash
# Use pyenv to create virtual environment:
git clone https://github.com/Dicklesworthstone/llm_aided_transcription_improvement  
cd llm_aided_transcription_improvement          
pyenv local 3.12
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

3. Set up your environment variables in a `.env` file:
   ```
   USE_LOCAL_LLM=False
   API_PROVIDER=OPENAI
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   CLAUDE_MODEL_STRING=claude-3-haiku-20240307
   OPENAI_COMPLETION_MODEL=gpt-4o-mini
   ```

## Usage

1. Place your transcription JSON file in the project directory.

2. Update the input file name in the `main()` function if necessary.

3. Run the script:
   ```
   python llm_aided_transcription_improvement.py
   ```

4. The script will generate the formatted transcription as `formatted_transcription.md`.

## How It Works

The LLM-Aided Transcription project employs a multi-step process to transform raw transcription output into high-quality, readable text:

1. **Chunk Creation**: Splits the raw transcription into manageable chunks for processing.

2. **Multi-Stage Processing**: Each chunk undergoes a three-stage LLM-based processing to correct errors, improve readability, and format the text.

3. **Parallel Processing**: Chunks are processed concurrently to improve speed when using API-based models.

4. **Context Preservation**: Each chunk includes a small overlap with adjacent chunks to maintain context.

5. **Quality Assessment**: An LLM-based evaluation compares the final output quality to the original transcription text.

## Code Optimization

- **Concurrent Processing**: When using API-based models, chunks are processed concurrently to improve speed.
- **Adaptive Token Management**: The system dynamically adjusts the number of tokens used for LLM requests based on input size and model constraints.
- **GPU Acceleration**: For local LLMs, the system attempts to use GPU acceleration when available.

## Limitations and Future Improvements

- The system's performance is heavily dependent on the quality of the LLM used.
- Processing very large transcriptions can be time-consuming and may require significant computational resources.
- Future improvements could include:
  - Support for more LLM providers
  - Enhanced error handling and recovery mechanisms
  - Integration with real-time transcription systems

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
