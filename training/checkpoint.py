import os
import torch

def save_checkpoint(
    model, 
    optimizer, 
    scheduler, 
    step, 
    epoch, 
    train_loss, 
    val_loss, 
    config, 
    tokenizer_info, 
    path
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        "tokenizer_info": tokenizer_info
    }
    # Save to a temporary file first, then rename to ensure atomic write (prevents corruption)
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)
    print(f"COLLISION-1M checkpoint saved successfully to {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location="cpu")
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if passed and available
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    # Load scheduler state if passed and available
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    print(f"Loaded checkpoint from {path} at step {checkpoint['step']}, epoch {checkpoint['epoch']}")
    return checkpoint
