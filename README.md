# M4A Audio Transcriber

This project provides a simple yet powerful command-line tool to transcribe M4A audio files into text using **OpenAI's Whisper ASR model**. It leverages your NVIDIA GPU (via CUDA on WSL2 for Windows users) for faster processing and handles batch transcription of multiple files.

-----

## Features

* **High-Quality Transcription:** Uses various Whisper models (tiny, base, small, medium, large) for accurate speech-to-text conversion.
* **GPU Accelerated (NVIDIA CUDA):** Automatically uses your NVIDIA GPU for faster transcription.
* **M4A Support:** Handles M4A audio files, converting them to WAV internally.
* **Batch Processing:** Transcribes all M4A files within a specified input directory.
* **Organized Output:** Saves transcriptions as `.txt` files, mirroring original audio names.
* **Customizable Model:** Easily select the desired Whisper model via command-line arguments.

-----

## Prerequisites

Before you begin, ensure you have the following:

1. **Python 3.8+**
2. **pip**
3. **FFmpeg:**
    * **On Ubuntu (WSL2):**
      ```bash
      sudo apt update
      sudo apt install ffmpeg -y
      ```
    * **On Windows:** Download from the [official FFmpeg website](https://ffmpeg.org/download.html) and add to your system's PATH.
4. **NVIDIA GPU with CUDA (Highly Recommended):**
    * For Windows users with WSL2, follow the official guide for [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl).
    * Verify PyTorch detects your GPU inside WSL:
      ```bash
      python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
      ```

-----

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/your_repo_name.git
    cd your_repo_name
    ```

2. **Create input and output directories:**
    ```bash
    mkdir audios_input
    mkdir transcripts_output
    ```

    Place your `.m4a` audio files into `audios_input/`.

3. **Create and activate a Python Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Linux/WSL
    ```

4. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have `requirements.txt`, install manually: `pip install pydub openai-whisper`)*

-----

## Usage

Navigate to your project directory in your terminal (with the virtual environment activated).

The script `transcriptor.py` takes the input directory as a required argument.

### Basic Usage (using default 'small' Whisper model)

# Transcriptor Documentation

## Uso básico

```bash
python3 transcriptor.py audios_input/
```

This transcribes all M4A files in `audios_input/` and saves `.txt` files in `transcripts_output/`.

## Specifying a different Whisper Model

Use the `--model` flag. Larger models offer higher accuracy but are slower.

**Available Models:** `tiny`, `base`, `small`, `medium`, `large`

**English-only Models:** `tiny.en`, `base.en`, `small.en`, `medium.en`

### Example using the medium model:

```bash
python3 transcriptor.py audios_input/ --model medium
```

### Example using the large model and a custom output directory:

```bash
python3 transcriptor.py audios_input/ --model large --output_dir my_custom_transcripts/
```

## Getting Help

```bash
python3 transcriptor.py --help
```

## How it Works

The script converts M4A to a temporary WAV, then uses the chosen Whisper model to transcribe the audio, and finally saves the text to a `.txt` file, cleaning up temporary files.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests.

## License

```
MIT License

Copyright (c) [2025] [M4A audio transcriber]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

