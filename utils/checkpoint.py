import os
import torch
import glob


def save_checkpoint(state, filepath):
    """
    Safely save checkpoint by writing to a temp file first, then renaming.
    Prevents corrupted checkpoints if the process is killed during write.
    """
    tmp_filepath = filepath + ".tmp"
    torch.save(state, tmp_filepath)
    os.replace(tmp_filepath, filepath)

def auto_resume_helper(output_dir):
    if not os.path.exists(output_dir):
        print(f"Save dir {output_dir} does not exist for auto-resume.")
        return None
        
    # Priority 1: last_model.pth
    last_ckpt = os.path.join(output_dir, "last_model.pth")
    if os.path.exists(last_ckpt):
        print(f"Auto-resume: found {last_ckpt}")
        return last_ckpt
        
    # Priority 2: ckpt_*.pth or any pth, sorted by mtime
    checkpoints = glob.glob(os.path.join(output_dir, "*.pth"))
    # Filter out best_model usually unless it's the only one, but standard practice is resuming from latest state
    checkpoints = [ckpt for ckpt in checkpoints if "best_model" not in ckpt]
    
    if len(checkpoints) > 0:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Auto-resume: found latest checkpoint {latest_checkpoint}")
        return latest_checkpoint
        
    return None

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None, model_ema=None):
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found: {filepath}")
        return 0, 1e10
        
    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location='cpu')

    # 1. Load Model
    # handle cases where key is 'net' (train.py) or 'model' (general)
    model_state = checkpoint.get('net', checkpoint.get('model', None))
    if model_state is not None:
        # Handle DDP 'module.' prefix if needed
        new_state_dict = {}
        for k, v in model_state.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        print("Warning: No model state found in checkpoint.")

    # 2. Load Optimizer
    if optimizer is not None:
        opt_state = checkpoint.get('optimizer_state', checkpoint.get('optimizer', None))
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    # 3. Load Scheduler
    if scheduler is not None:
        sched_state = checkpoint.get('scheduler_state', checkpoint.get('lr_scheduler', None))
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

    # 4. Load Scaler
    if scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])

    # 5. Load EMA
    if model_ema is not None:
        ema_state = checkpoint.get('net_ema', checkpoint.get('model_ema', None))
        if ema_state is not None:
            model_ema.load_state_dict(ema_state)
            print("Loaded EMA state.")
        else:
            print("Warning: EMA state not found in checkpoint.")

    # 6. Metadata
    start_epoch = checkpoint.get('epoch', -1) + 1
    min_loss = checkpoint.get('min_loss', 1e10)
    
    # If config overrides are needed, they are typically handled outside
    print(f"Loaded successfully. Resuming from epoch {start_epoch}, previous min_loss {min_loss:.6f}")

    return start_epoch, min_loss