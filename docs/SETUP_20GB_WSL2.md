# Running AdaptThink evaluation on one 20 GB GPU (WSL2 Ubuntu)

Tested on: RTX A4500 (20 GB), Windows driver 596.51, WSL2 Ubuntu, Python 3.10, torch 2.6.0+cu124, vLLM 0.8.2.
The upstream scripts target 8x A100-80GB; the steps below are what was needed on a single 20 GB card.

## 1. Environment

    conda create -n adapt_think python=3.10 -y
    conda activate adapt_think
    git clone https://github.com/THU-KEG/AdaptThink.git && cd AdaptThink
    pip install uv
    echo "xgrammar==0.1.17" > overrides.txt
    uv pip install --python $(which python) -r requirements.txt --override overrides.txt
    pip install math-verify
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

Check: `python -c "import torch, vllm, flash_attn; print(torch.cuda.is_available(), vllm.__version__)"`

## 2. Models and data

    huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ~/models/DeepSeek-R1-Distill-Qwen-1.5B
    huggingface-cli download THU-KEG/AdaptThink-1.5B-delta0.05 --local-dir ~/models/AdaptThink-1.5B-delta0.05
    bash scripts/preprocess_dataset.sh

The test sets are already in `data/test/` in the upstream repo. Both models' chat templates must end with `<｜Assistant｜><think>\n`.

## 3. Run

`scripts/run_eval_local.sh` evaluates both models. Choose datasets with `VAL_SETS`:

    VAL_SETS="gsm8k" bash scripts/run_eval_local.sh
    VAL_SETS="math" bash scripts/run_eval_local.sh
    VAL_SETS="aime*16" bash scripts/run_eval_local.sh     # 16 samples per problem

Results are written to `outputs/<model>/eval_results_<dataset>_len16384_n1/` (`overall_metrics.json` and `data/*.json`).
Observed wall-clock for both models: GSM8K ~1 h, MATH500 ~2 h, AIME24 ~9.5 h.

## 4. Problems we hit and what fixed them

| Symptom | Fix |
|---|---|
| `CondaToSNonInteractiveError` | `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` (and `.../pkgs/r`) |
| `pip install -r requirements.txt` fails: `xgrammar==0.1.16` not found | Install with `uv` and `--override` pinning `xgrammar==0.1.17` |
| `No module named 'math_verify'` | `pip install math-verify` |
| `No module named 'flash_attn'` | Prebuilt wheel above (needs `torch._C._GLIBCXX_USE_CXX11_ABI == False`) |
| `max_num_batched_tokens is smaller than max_model_len` | max_model_len = 4096 + 16384 = 20480, so use `max_num_batched_tokens=24576` |
| `No available memory for the cache blocks` (at gpu_memory_utilization 0.5, 0.7, 0.8) | `gpu_memory_utilization=0.9` and `max_num_seqs=64`, plus the FSDP offload flags in `run_eval_local.sh` |
| `Total available GPUs 1 is less than total desired GPUs 8` | A blank line inside the multi-line command ended it early, so the remaining overrides were never applied. No blank lines between `\` continuations |
| Two runs started by mistake | `pkill -f run_eval_local.sh; pkill -f main_ppo; ray stop --force`, then check `nvidia-smi` |
| Paste does not work in WSL terminal | `Ctrl+Shift+V`, or use VS Code with the WSL extension |

The GPU memory settings are what worked on this machine; I did not isolate which single setting was the decisive one.

## 5. Evaluating other checkpoints (delta sweep)

`run_eval_local.sh` takes a space-separated list of model directories in `MODELS`
(default: original DeepSeek-R1-Distill-Qwen-1.5B and AdaptThink-1.5B-delta0.05):

    M="$HOME/models/AdaptThink-1.5B-delta0 $HOME/models/AdaptThink-1.5B-delta0.01"
    MODELS="$M" VAL_SETS="gsm8k" bash scripts/run_eval_local.sh

Then `python analysis/delta_sweep.py` builds the table in `results/delta_sweep.md`.
Wall-clock on the RTX A4500: GSM8K plus MATH500 for the five extra checkpoints took roughly 3.5 hours in total.

## 6. Demo app (Gradio)

The app compares the original AdaptThink-1.5B (delta 0.05) with the LoRA-tuned checkpoints on the same question.
It needs its own environment, because installing Gradio in `adapt_think` can change vLLM's pinned packages:

    conda create -n adapt_gui python=3.10 -y && conda activate adapt_gui
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install transformers==4.51.2 accelerate gradio
    python app/app.py        # then open http://localhost:7860

It loads each model (about 3.5 GB) on the GPU, so do not run it while a training or vLLM evaluation job is using the same card.
