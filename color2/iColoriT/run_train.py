import subprocess
import os

def run_training(gpu_id=0, master_port=4885, extra_args=""):
    """
    Runs the iColoriT training script.

    Args:
        gpu_id (int): The CUDA device ID to use.
        master_port (int): The master port for distributed training.
        extra_args (str): Any additional arguments to pass to train_new.py.
    """
    # === Path Setup ===
    data_path = "/home/sapanagupta/ICOLORIT_INPUTS/patch_based_process/Train/imgs"
    val_data_path = "/home/sapanagupta/ICOLORIT_INPUTS/patch_based_process/Valid/imgs"
    output_dir = "/home/sapanagupta/ICOLORIT_INPUTS/patch_colorized"
    tb_log_dir = "/home/sapanagupta/ICOLORIT_INPUTS/patch_colorized/log/"
    pretrained_weights = "/home/sapanagupta/ICOLORIT_INPUTS/MODELS/icolorit_base_4ch_patch16_224.pth"
    train_script_path = "/home/sapanagupta/PycharmProjects/color2/iColoriT/train_new.py"

    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MASTER_PORT"] = str(master_port)

    # Construct the command
    command = [
        "torchrun",
        "--nproc_per_node=1",
        f"--master_port={master_port}",
        train_script_path,
        "--data_path", data_path,
        "--val_data_path", val_data_path,
        "--output_dir", output_dir,
        "--log_dir", tb_log_dir,
        "--exp_name", "exp_finetune_maskhint",
        "--save_args_txt",
        "--batch_size=32",
        "--lr=0.00005",
        "--num_workers=16",
        "--pin_mem",
        "--epochs=50",
        "--resume", pretrained_weights,
        "--hint_size", "4",
        "--resume_weights_only",
        "--no_avg_hint",
    ]

    # Add any extra arguments
    if extra_args:
        command.extend(extra_args.split())

    print(f"Executing command: {' '.join(command)}")
    try:
        # Use subprocess.run for better control and error handling
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
    except FileNotFoundError:
        print("Error: 'torchrun' command not found. Make sure PyTorch distributed is installed and in your PATH.")

if __name__ == "__main__":
    # You can call run_training with specific parameters or leave defaults
    run_training(gpu_id=0, master_port=4885)
    # To pass additional arguments, for example, if you want to override epochs:
    # run_training(extra_args="--epochs 10")