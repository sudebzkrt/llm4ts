import argparse
import torch
import numpy as np
from copy import deepcopy
import shutil
from pathlib import Path

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_supervised_finetuning import Exp_Supervised_Finetuning
from utils.tools import set_seed, print_formatted_dict
from data_provider.data_loader import update_args_from_dataset


def get_args_from_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM4TS")

    # --- ORİJİNAL ARGÜMANLAR ---
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--model_id", type=str, default="test")
    parser.add_argument("--model", type=str, default="LLM4TS")

    # ÖNEMLİ: fixed/tunable parametrelerin uygulanması için TRUE
    parser.add_argument("--overwrite_args", action="store_true", default=True)
    parser.add_argument("--delete_checkpoints", action="store_true", default=False)

    # CUSTOM STOCK'u DEFAULT YAPIYORUZ
    parser.add_argument("--data_name", type=str, default="custom_stock")
    parser.add_argument("--data", type=str, default="custom_stock")
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="CO1/co1_comdty.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="co1_comdty")
    parser.add_argument("--freq", type=str, default="d")
    parser.add_argument(
        "--checkpoints", type=str, default="./checkpoints/"
    )
    parser.add_argument(
        "--pred_len_list", type=int, nargs="+", default=[30]
    )
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--label_len", type=int, default=64)
    parser.add_argument("--pred_len", type=int, default=30)
    parser.add_argument("--LLM", type=str, default="gpt2", choices=["gpt2"])
    parser.add_argument("--no_pretrain", action="store_true", default=False)
    parser.add_argument("--no_freeze", action="store_true", default=False)

    parser.add_argument("--first_k_layers", type=int, default=6)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument(
        "--token_embed_type", type=str, default="conv", choices=["linear", "conv"]
    )
    parser.add_argument("--token_embed_kernel_size", type=int, default=3)
    parser.add_argument(
        "--temporal_embed_type",
        type=str,
        default="learned",
        choices=["none", "fixed", "learned", "timeF"],
    )
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--peft_method", type=str, default="lora", choices=["none", "lora", "adalora"]
    )
    parser.add_argument("--peft_params_r", type=int, default=4)
    parser.add_argument("--peft_params_lora_alpha", type=int, default=32)
    parser.add_argument("--peft_params_lora_dropout", type=float, default=0.2)
    parser.add_argument(
        "--enable_supervised_finetuning", type=bool, default=True
    )
    parser.add_argument(
        "--sft_optim", type=str, default="Adam", choices=["Adam", "AdamW", "RMSprop"]
    )
    parser.add_argument("--sft_learning_rate", type=float, default=0.001)
    parser.add_argument("--sft_lradj", type=str, default="type1")
    parser.add_argument("--sft_weight_decay", type=float, default=0.001)
    parser.add_argument("--sft_train_epochs", type=int, default=10)
    parser.add_argument(
        "--dft_optim", type=str, default="Adam", choices=["Adam", "AdamW", "RMSprop"]
    )
    parser.add_argument("--dft_learning_rate", type=float, default=0.001)
    parser.add_argument("--dft_lradj", type=str, default="type1")
    parser.add_argument("--dft_weight_decay", type=float, default=0.001)
    parser.add_argument("--dft_train_epochs", type=int, default=10)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.0001)
    parser.add_argument(
        "--ft_mode", type=str, default="lp_ft", choices=["lp_ft", "lp", "ft"]
    )
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true", default=True)

    args, _ = parser.parse_known_args()

    # DEVICE SEÇİMİ
    if args.use_gpu:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.use_gpu = False
            args.device = "cpu"
    else:
        args.device = "cpu"

    args.root_path = Path.cwd()
    args.return_single_feature = True
    return args


def update_args_from_fixed_params(
    args: argparse.Namespace, fixed_params: dict
) -> argparse.Namespace:
    for key, value in fixed_params.items():
        print(f"### [Fixed] Set {key} to {value}")
        setattr(args, key, value)
    return args


def update_args_from_tunable_params(
    args: argparse.Namespace, tunable_params: dict
) -> argparse.Namespace:
    for key, value in tunable_params.items():
        print(f"### [Tunable] Set {key} to {value}")
        setattr(args, key, value)
    return args


def update_args(args, fixed_params, tunable_params):
    duplicated_keys = set(fixed_params.keys()) & set(tunable_params.keys())
    assert not duplicated_keys, f"Duplicated keys found: {duplicated_keys}"

    # overwrite_args TRUE ise fixed + tunable parametreler uygulanır
    if args.overwrite_args:
        args = update_args_from_fixed_params(args, fixed_params)
        args = update_args_from_tunable_params(args, tunable_params)

    # Dataset'e göre ek arg güncellemesi
    args = update_args_from_dataset(args)

    args.setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_eb{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.embed,
    )
    print(f"Args in experiment: {args}")

    # SFT argümanları
    sft_args = deepcopy(args)
    sft_args.task_name = "supervised_finetuning"
    sft_args.ft_mode = "ft"
    sft_args.features = "M"
    sft_args.pred_len = sft_args.stride
    sft_args.label_len = sft_args.seq_len - sft_args.pred_len
    sft_args.setting = "sft_" + sft_args.setting

    dft_args = deepcopy(args)
    return sft_args, dft_args


def get_exp(args):
    if args.task_name == "long_term_forecast":
        exp = Exp_Long_Term_Forecast(args)
    elif args.task_name == "supervised_finetuning":
        exp = Exp_Supervised_Finetuning(args)
    else:
        raise NotImplementedError
    return exp


def trainable(tunable_params: dict, fixed_params: dict, args: argparse.Namespace) -> dict:
    sft_args, dft_args = update_args(args, fixed_params, tunable_params)

    # Önce SFT (Supervised Fine-Tuning)
    if dft_args.enable_supervised_finetuning:
        sft_exp = get_exp(sft_args)
        sft_metrics = sft_exp.train(use_tqdm=True)
        print("SFT metrics:", sft_metrics)

    # Sonra DFT (Direct Forecasting)
    dft_metrics_dict = {}
    for pred_len in dft_args.pred_len_list:
        dft_args.pred_len = pred_len
        dft_exp = get_exp(dft_args)
        dft_metrics_dict[pred_len] = dft_exp.train(use_tqdm=True)

    return_metrics = {}
    return_metrics["avg_mse"] = np.mean(
        [v["best_test_loss"] for v in dft_metrics_dict.values()]
    )
    return_metrics["avg_mae"] = np.mean(
        [v["best_test_mae"] for v in dft_metrics_dict.values()]
    )
    for pred_len in dft_args.pred_len_list:
        return_metrics[f"{pred_len}_mse"] = dft_metrics_dict[pred_len][
            "best_test_loss"
        ]
        return_metrics[f"{pred_len}_mae"] = dft_metrics_dict[pred_len][
            "best_test_mae"
        ]

    if args.delete_checkpoints:
        shutil.rmtree(args.checkpoints)

    return return_metrics


if __name__ == "__main__":
    set_seed(seed=2023)
    args = get_args_from_parser()

    # === CUSTOM STOCK BLOĞU (CO1 / co1_comdty.csv) ===
    if args.data == "custom_stock":
        print(">>> KENDİ VERİ SETİ AYARLARI (custom_stock) YÜKLENİYOR...")

        from data_provider.data_factory import data_provider

        temp_dataset, _ = data_provider(args, flag="train")
        args.enc_in = temp_dataset.C
        print(f">>> Otomatik feature sayısı (enc_in): {args.enc_in}")

        # Sabit parametreler
        fixed_params = {
            "data_name": "custom_stock",
            "pred_len_list": [30],  # TEK HORIZON: 30 GÜNLÜK TAHMİN
            "percent": 100,
            "num_workers": 8,
            "batch_size": args.batch_size,
            "root_path": args.root_path,
            "data_path": args.data_path,
            "enc_in": args.enc_in,
            "target": args.target,
            "freq": "d",
        }

        # Ayarlanabilir parametreler
        tunable_params = {
            "enable_supervised_finetuning": True,
            "first_k_layers": 6,
            "patch_len": args.patch_len,
            "stride": args.stride,
            "seq_len": args.seq_len,
            "ft_mode": "lp_ft",
            "dropout": args.dropout,
            "token_embed_type": "conv",
            "token_embed_kernel_size": 3,
            "temporal_embed_type": "learned",
            "peft_method": "lora",
            "sft_optim": "AdamW",
            "sft_learning_rate": args.sft_learning_rate,
            "sft_lradj": "constant",
            "sft_weight_decay": 0.0,
            "sft_train_epochs": args.train_epochs,
            "dft_optim": "AdamW",
            "dft_learning_rate": args.dft_learning_rate,
            "dft_lradj": "constant",
            "dft_weight_decay": 0.0,
            "dft_train_epochs": args.train_epochs,
            "peft_params_r": args.peft_params_r,
            "peft_params_lora_alpha": 32,
            "peft_params_lora_dropout": args.peft_params_lora_dropout,
        }

    else:
        # ORİJİNAL ETT AYARLARI (istersen sonradan kullanırsın)
        print(f">>> VARSAYILAN AYARLAR ({args.data_name}) YÜKLENİYOR...")
        fixed_params = {
            "data_name": args.data_name,
            "pred_len_list": [96],
            "percent": 100,
            "num_workers": 8,
            "batch_size": 128,
        }
        tunable_params = {
            "enable_supervised_finetuning": True,
            "first_k_layers": 6,
            "patch_len": 16,
            "stride": 8,
            "seq_len": 336,
            "ft_mode": "lp_ft",
            "dropout": 0.05,
            "token_embed_type": "conv",
            "token_embed_kernel_size": 3,
            "temporal_embed_type": "learned",
            "freq": "h",
            "peft_method": "adalora",
            "sft_optim": "AdamW",
            "sft_learning_rate": 7.912045141879411e-05,
            "sft_lradj": "constant",
            "sft_weight_decay": 0.0005542494992024964,
            "sft_train_epochs": 5,
            "dft_optim": "AdamW",
            "dft_learning_rate": 1.8257759510439175e-05,
            "dft_lradj": "constant",
            "dft_weight_decay": 0.0014555863788252605,
            "dft_train_epochs": 15,
            "peft_params_r": 8,
            "peft_params_lora_alpha": 64,
            "peft_params_lora_dropout": 0,
        }

    return_metrics = trainable(tunable_params, fixed_params, args)
    print_formatted_dict(return_metrics)
