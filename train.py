# train.py
from llamafactory.train import run_exp   # LLaMA-Factory 内部入口
from llamafactory.hparams import get_train_args  # 解析训练参数
import os

# 1. 将命令行参数写成 Python 字典
train_args = [
    "--stage", "sft",
    "--do_train",
    "--model_name_or_path", "/root/autodl-tmp/miniCPM/models/OpenBMB/MiniCPM-o-2_6",
    "--preprocessing_num_workers", "16",
    "--finetuning_type", "lora",
    "--template", "minicpm_o",
    "--flash_attn", "auto",
    "--dataset_dir", "/root/autodl-tmp/LLaMA-Factory/data/",
    "--dataset", "e_f",
    "--cutoff_len", "2048",
    "--learning_rate", "5e-05",
    "--num_train_epochs", "5.0",
    "--max_samples", "100000",
    "--per_device_train_batch_size", "2",
    "--gradient_accumulation_steps", "8",
    "--lr_scheduler_type", "cosine",
    "--max_grad_norm", "1.0",
    "--logging_steps", "5",
    "--save_steps", "100",
    "--warmup_steps", "0",
    "--output_dir", "saves/MiniCPM-o-2_6/lora/train_2025-07-18-16-18-32",
    "--bf16",
    "--plot_loss",
    "--trust_remote_code",
    "--ddp_timeout", "180000000",
    "--include_num_input_tokens_seen",
    "--optim", "adamw_torch",
    "--lora_rank", "8",
    "--lora_alpha", "16",
    "--lora_dropout", "0.2",
    "--lora_target", "all",
    "--additional_target", "v_proj,k_proj",
]

# 2. 让 LLaMA-Factory 解析并运行
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(train_args)
run_exp(model_args, data_args, training_args, finetuning_args, generating_args)