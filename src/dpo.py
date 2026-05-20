from functools import partial
import json
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from architecture import GPTModel
from pretrain import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    load_weights_into_gpt,
    generate_and_print_sample,
)
from gpt_download import download_and_load_gpt2
from dpo_dataset import PreferenceDataset, custom_collate_fn, format_input


# =========================================================
# DPO Loss Computation
# =========================================================

def compute_logprobs(logits, labels, selection_mask=None):
    
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    # Convert raw logits to log probabilities across the vocabulary
    log_probs = F.log_softmax(logits, dim=-1)

    # Pick the log-prob of the actual next token at each position
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)  

    if selection_mask is not None:
        # Shift the mask to match the shifted labels
        mask = selection_mask[:, 1:].clone()

        # Zero out log-probs for tokens we want to ignore (prompt / padding)
        selected_log_probs = selected_log_probs * mask

        # Average over the valid (non-masked) tokens per sequence
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)
        return avg_log_prob
    else:
        return selected_log_probs.mean(-1)


def compute_dpo_loss(
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
    beta=0.1,
):
    
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):

    # Policy model 
    policy_chosen_logprobs = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"],
    )
    policy_rejected_logprobs = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"],
    )

    # Reference model 
    with torch.no_grad():
        ref_chosen_logprobs = compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"],
        )
        ref_rejected_logprobs = compute_logprobs(
            logits=reference_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"],
        )

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_logprobs,
        model_rejected_logprobs=policy_rejected_logprobs,
        reference_chosen_logprobs=ref_chosen_logprobs,
        reference_rejected_logprobs=ref_rejected_logprobs,
        beta=beta,
    )
    return loss, chosen_rewards, rejected_rewards


def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):

    total_loss, total_chosen_rewards, total_rejected_rewards = 0.0, 0.0, 0.0

    if len(data_loader) == 0:
        return float("nan"), float("nan"), float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta,
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()
        else:
            break

    return (
        total_loss / num_batches,
        total_chosen_rewards / num_batches,
        total_rejected_rewards / num_batches,
    )


def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):

    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_r, train_rejected_r = compute_dpo_loss_loader(
            train_loader, policy_model, reference_model, beta, num_batches=eval_iter
        )
        val_loss, val_chosen_r, val_rejected_r = compute_dpo_loss_loader(
            val_loader, policy_model, reference_model, beta, num_batches=eval_iter
        )

    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_r,
        "train_rejected_reward": train_rejected_r,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_r,
        "val_rejected_reward": val_rejected_r,
    }
    policy_model.train()
    return res


# =========================================================
# DPO Training Loop
# =========================================================

def train_model_dpo_simple(
    policy_model, reference_model, train_loader, val_loader,
    optimizer, num_epochs, beta, eval_freq, eval_iter,
    start_context, tokenizer,
):
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": [],
    }
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        policy_model.train()

        for batch in train_loader:
            optimizer.zero_grad()

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta,
            )

            loss.backward()
            optimizer.step()

            tokens_seen += batch["chosen"].numel()
            global_step += 1

            if global_step % eval_freq == 0:
                res = evaluate_dpo_loss_loader(
                    policy_model, reference_model, train_loader, val_loader, beta, eval_iter
                )
                
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)

                train_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_margin = res["val_chosen_reward"] - res["val_rejected_reward"]
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margin {train_margin:.3f}, "
                    f"Val reward margin {val_margin:.3f}"
                )

        # Print a sample after each epoch
        generate_and_print_sample(
            model=policy_model,
            tokenizer=tokenizer,
            device=start_context.device if isinstance(start_context, torch.Tensor) else "cpu",
            start_context=start_context,
        )

    return tracking


# =========================================================
# Plotting
# =========================================================

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, label="loss"):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs_seen, train_losses, label=f"Train {label}")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label=f"Val {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label)
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plot_name = f"figures/dpo-{label}.pdf"
    plt.savefig(plot_name)
    print(f"Plot saved as {plot_name}")


# =========================================================
# Main
# =========================================================

def main(test_mode=False):
    # --------------------------------------------------
    #  Load preference dataset
    # --------------------------------------------------
    file_path = "data/instruction-data-with-preference.json"
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    if test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    # --------------------------------------------------
    #  Set up device and tokenizer
    # --------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        device = torch.device("mps") if (major, minor) >= (2, 9) else torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        mask_prompt_tokens=True,
        allowed_max_length=1024,
    )

    # --------------------------------------------------
    #  Create data loaders
    # --------------------------------------------------
    batch_size = 8 if not test_mode else 2
    num_workers = 0
    torch.manual_seed(123)

    train_dataset = PreferenceDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
        shuffle=True, drop_last=True, num_workers=num_workers,
    )

    val_dataset = PreferenceDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
        shuffle=False, drop_last=False, num_workers=num_workers,
    )

    # --------------------------------------------------
    #  Load instruction-finetuned model 
    # --------------------------------------------------
    if test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2,
        }
        policy_model = GPTModel(BASE_CONFIG)
        policy_model.eval()
        device = torch.device("cpu")
        CHOOSE_MODEL = "Small test model"

    else:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.0,
            "qkv_bias": True,
        }
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }
        CHOOSE_MODEL = "gpt2-medium (355M)"
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        # Load SFT-finetuned weights
        sft_checkpoint = "gpt2-medium355M-sft.pth"
        if not os.path.exists(sft_checkpoint):
            print(f"WARNING: {sft_checkpoint} not found. Loading pretrained GPT-2 (not SFT).")
            print("For best results, run finetuning.py first to create the SFT checkpoint.")
            model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
            settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
            policy_model = GPTModel(BASE_CONFIG)
            load_weights_into_gpt(policy_model, params)
        else:
            policy_model = GPTModel(BASE_CONFIG)
            policy_model.load_state_dict(
                torch.load(sft_checkpoint, map_location="cpu", weights_only=True)
            )

        policy_model.eval()
        policy_model.to(device)

    print("Policy model:", CHOOSE_MODEL)

    # --------------------------------------------------
    #  Create frozen reference model (copy of SFT model)
    # --------------------------------------------------
    reference_model = GPTModel(BASE_CONFIG)
    if test_mode:
        pass  
    else:
        if os.path.exists(sft_checkpoint):
            reference_model.load_state_dict(
                torch.load(sft_checkpoint, map_location="cpu", weights_only=True)
            )
        else:
            reference_model.load_state_dict(policy_model.state_dict())

    reference_model.eval()
    reference_model.to(device)
    # Freeze all parameters 
    for param in reference_model.parameters():
        param.requires_grad = False

    print("Reference model loaded and frozen.")

    # --------------------------------------------------
    #  Initial losses and rewards
    # --------------------------------------------------
    beta = 0.1
    res = evaluate_dpo_loss_loader(
        policy_model, reference_model, train_loader, val_loader, beta, eval_iter=5
    )
    print("\nInitial DPO metrics:")
    print(f"  Train loss: {res['train_loss']:.3f}, Val loss: {res['val_loss']:.3f}")
    print(f"  Train reward margin: {res['train_chosen_reward'] - res['train_rejected_reward']:.3f}")
    print(f"  Val reward margin:   {res['val_chosen_reward'] - res['val_rejected_reward']:.3f}")

    # --------------------------------------------------
    #  Train with DPO
    # --------------------------------------------------
    start_time = time.time()

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)
    num_epochs = 1  

    torch.manual_seed(123)
    tracking = train_model_dpo_simple(
        policy_model=policy_model,
        reference_model=reference_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        beta=beta,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )

    execution_time_minutes = (time.time() - start_time) / 60
    print(f"\nTraining completed in {execution_time_minutes:.2f} minutes.")

    # --------------------------------------------------
    #  Plot results
    # --------------------------------------------------
    epochs_tensor = torch.linspace(0, num_epochs, len(tracking["train_losses"]))

    plot_losses(epochs_tensor, tracking["tokens_seen"],
                tracking["train_losses"], tracking["val_losses"], label="loss")

    train_margins = [c - r for c, r in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
    val_margins = [c - r for c, r in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]
    plot_losses(epochs_tensor, tracking["tokens_seen"], train_margins, val_margins, label="reward-margins")

    # --------------------------------------------------
    #  Generate sample responses (reference vs policy)
    # --------------------------------------------------
    print("\n=== Sample responses (reference vs policy) ===\n")
    torch.manual_seed(123)

    for entry in test_data[:3]:
        input_text = format_input(entry)

        # Reference model response
        ref_ids = generate(
            model=reference_model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        ref_response = token_ids_to_text(ref_ids, tokenizer)[len(input_text):].replace("### Response:", "").strip()

        # Policy model response (after DPO)
        pol_ids = generate(
            model=policy_model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        pol_response = token_ids_to_text(pol_ids, tokenizer)[len(input_text):].replace("### Response:", "").strip()

        print(f"Instruction: {entry['instruction']}")
        print(f"  Reference: {ref_response}")
        print(f"  Policy:    {pol_response}")
        print(f"  Chosen:    {entry['chosen']}")
        print(f"  Rejected:  {entry['rejected']}")
        print("-" * 60)

    # --------------------------------------------------
    #  Save model
    # --------------------------------------------------
    model_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-dpo.pth"
    torch.save(policy_model.state_dict(), model_name)
    print(f"\nModel saved as {model_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finetune a GPT model with DPO")
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help="Run with a tiny model for quick testing",
    )
    args = parser.parse_args()

    main(args.test_mode)