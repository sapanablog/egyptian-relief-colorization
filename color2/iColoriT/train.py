# #''''
# import argparse
# import datetime
# import json
# import os
# import random
# import time
#
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# from timm.models import create_model
# from torch.optim import optimizer
# from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
#
# import modeling  # To register models
# import utils
# from datasets_org_till11thjune import build_pretraining_dataset, build_fixed_validation_dataset, build_validation_dataset
# from engine import train_one_epoch, validate
# from optim_factory import create_optimizer
# from utils import NativeScalerWithGradNormCount as NativeScaler
#
# import sys
#
# # --- Backfill utils.save_args if it's missing ---
# if not hasattr(utils, "save_args"):
#     def _save_args_wrapper(args, out_dir, save_pkl=True, save_txt=True):
#         # delegate to your local fallback
#         _save_args_fallback(args, out_dir, save_pkl=save_pkl, save_txt=save_txt)
#     utils.save_args = _save_args_wrapper
#
# # --- add to train.py ---
# def _save_args_fallback(args, out_dir, save_pkl=True, save_txt=True):
#     import json, os, pickle, datetime
#     os.makedirs(out_dir, exist_ok=True)
#     args_dict = vars(args)
#
#     # JSON
#     with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
#         json.dump(args_dict, f, indent=2, default=str)
#
#     # TXT (pretty)
#     if save_txt:
#         ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as f:
#             f.write(f"# iColoriT args saved at {ts}\n")
#             for k in sorted(args_dict.keys()):
#                 f.write(f"{k}: {args_dict[k]}\n")
#
#     # PKL (optional)
#     if save_pkl:
#         with open(os.path.join(out_dir, "args.pkl"), "wb") as f:
#             pickle.dump(args, f)
#
# class TeeLogger:
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#
#     def write(self, message):
#         self.terminal.write(message)  # Print to terminal
#         self.log.write(message)       # Write to log file
#         self.flush()
#
#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()
#
# # Redirect stdout and stderr
# #sys.stdout = TeeLogger(f"/media/sapanagupta/vol1/Sapana/data/temp/training_log_462_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
# #sys.stdout = TeeLogger(f"//home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Train/train_log{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
# sys.stdout = TeeLogger(f"/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/train_ablation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
# #/media/sapanagupta/vol1/Sapana/data/Threshold/training_log_462_threshold
# sys.stderr = sys.stdout
#
# def get_args():
#     parser = argparse.ArgumentParser('iColoriT training scripts', add_help=False)
#     # Training
#     parser.add_argument('--exp_name', default='', type=str)
#     parser.add_argument('--epochs', default=100, type=int)
#     parser.add_argument('--batch_size', default=32, type=int)
#     parser.add_argument('--num_workers', default=8, type=int)
#     parser.add_argument('--save_ckpt_freq', default=5, type=int)
#     parser.add_argument('--seed', default=4885, type=int)
#     parser.add_argument('--save_args_pkl', action='store_true', help='Save args as pickle file')
#     parser.add_argument('--no_save_args_pkl', action='store_false', dest='save_args_pkl', help='')
#     parser.set_defaults(save_args_pkl=True)
#     parser.add_argument('--save_args_txt', action='store_true', help='Save args as txt file')
#     # Dataset (NOTE: these names must match dataset builders)
#     parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
#     parser.add_argument('--data_path', default='data/train/images', type=str, help='dataset path (TRAIN images)')
#     parser.add_argument('--val_data_path', default='data/val/images', type=str, help='validation dataset path')
#     parser.add_argument('--val_hint_dir', type=str, default='data/hint', help='base dir for fixed validation hints (contains h2-n*)')
#     # Fixed-train-hint support (your patch-wise TXT hints)
#     parser.add_argument('--train_hint_base_dir', type=str, default=None, help='Base dir of TRAIN hints; contains h2-n* subfolders')
#     parser.add_argument('--train_num_hint', type=int, default=None, help='Number of hints to use for training; picks subdir h2-n{N}')
#     # IO / resume
#     parser.add_argument('--output_dir', default='', help='path to save outputs')
#     parser.add_argument('--log_dir', default='tf_log', help='tensorboard log dir')
#     parser.add_argument('--resume', default='', help='checkpoint path')
#     parser.add_argument('--force_resume', action='store_true')
#     parser.add_argument('--start_epoch', default=0, type=int)
#     parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
#     parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
#     parser.set_defaults(pin_mem=True)
#     parser.add_argument('--gray_file_list_txt', type=str, default='', help='exclude these (optional)')
#     parser.add_argument('--return_name', action='store_true')
#     # Model
#     parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str)
#     parser.add_argument('--use_rpb', action='store_true')
#     parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
#     parser.set_defaults(use_rpb=True)
#     parser.add_argument('--head_mode', type=str, default='cnn', choices=['linear', 'cnn', 'locattn'])
#     parser.add_argument('--drop_path', type=float, default=0.0)
#     parser.add_argument('--mask_cent', action='store_true')
#     parser.add_argument('--avg_hint', action='store_true')
#     parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
#     parser.set_defaults(avg_hint=True)
#     # Hints
#     parser.add_argument('--hint_generator', type=str, default='RandomHintGenerator')
#     parser.add_argument('--num_hint_range', default=[0, 128], type=int, nargs=2)
#     parser.add_argument('--hint_size', default=4, type=int)
#     parser.add_argument('--val_hint_list', default=[1, 10, 20], nargs='+', type=int)
#     # LR / optim
#     parser.add_argument('--lr', type=float, default=5e-4)
#     parser.add_argument('--warmup_lr', type=float, default=1e-6)
#     parser.add_argument('--warmup_epochs', type=int, default=3)
#     parser.add_argument('--warmup_steps', type=int, default=-1)
#     parser.add_argument('--min_lr', type=float, default=1e-5)
#     parser.add_argument('--opt', default='adamw', type=str)
#     parser.add_argument('--opt_eps', default=1e-8, type=float)
#     parser.add_argument('--opt_betas', default=(0.9, 0.95), type=float, nargs='+')
#     parser.add_argument('--clip_grad', type=float, default=None)
#     parser.add_argument('--momentum', type=float, default=0.9)
#     parser.add_argument('--weight_decay', type=float, default=0.05)
#     parser.add_argument('--weight_decay_end', type=float, default=None)
#     # distributed
#     parser.add_argument('--device', default='cuda')
#     parser.add_argument('--world_size', default=1, type=int)
#     parser.add_argument('--local_rank', default=int(os.environ.get('LOCAL_RANK', 0)), type=int)
#     parser.add_argument('--dist_on_itp', action='store_true')
#     parser.add_argument('--dist_url', default='env://')
#     parser.add_argument('--auto_resume', action='store_true')
#     parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
#     parser.add_argument('--resume_weights_only', action='store_true', help='Only load model weights (skip optimizer)')
#     parser.set_defaults(auto_resume=True)
#     return parser.parse_args()
#
#
# # # # Use this to train the model from SCRATCH
# # def get_model(args):
# #     print(f"Creating model: {args.model}")
# #     model = create_model(
# #         args.model,
# #         pretrained=False,
# #         drop_path_rate=args.drop_path,
# #         drop_block_rate=None,
# #         use_rpb=args.use_rpb,
# #         avg_hint=args.avg_hint,
# #         head_mode=args.head_mode,
# #         mask_cent=args.mask_cent,
# #     )
# #     return model
#
# # Modified to run fine-tune model
# def get_model(args):
#     print(f"Creating model: {args.model}")
#     model = create_model(
#         args.model,
#         pretrained=True,  # Load pre-trained weights
#         drop_path_rate=args.drop_path,
#         drop_block_rate=None,
#         use_rpb=args.use_rpb,
#         avg_hint=args.avg_hint,
#         head_mode=args.head_mode,
#         mask_cent=args.mask_cent,
#     )
#
#     if args.resume:
#         checkpoint = torch.load(args.resume, map_location='cpu')
#         model.load_state_dict(checkpoint['model'], strict=False)
#         print(f"Loaded model weights from {args.resume}")
#
#         # Check if the flag --resume_weights_only is set
#         if not getattr(args, 'resume_weights_only', False):
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             print("Loaded optimizer state from checkpoint.")
#         else:
#             print("Skipping optimizer state loading (resume_weights_only flag is set).")
#
#     # Freeze all layers except the classification head (fine-tuning)
#     for param in model.parameters():
#         param.requires_grad = False
#
#         # # Freeze all layers except the classification head (fine-tuning)
#         # for param in model.parameters():
#         #     param.requires_grad = True
#
#     #for param in model.head.parameters():
#        # param.requires_grad = True  # Unfreeze the head
#
#     # Unfreeze last transformer blocks and head
#     for name, param in model.named_parameters():
#         if 'blocks.10' in name or 'blocks.11' in name or 'head' in name:
#             param.requires_grad = True
#
#     # If you want to gradually unfreeze the patch embedding layer or norm layers:
#     # for param in model.patch_embed.parameters():
#     #     param.requires_grad = True  # Unfreeze patch embedding
#     # for param in model.norm.parameters():
#     #     param.requires_grad = True  # Unfreeze final layer norm
#
#     print("Model fine-tuning enabled. All layers frozen except the head.")
#     return model
#
#
# def main(args):
#     utils.init_distributed_mode(args)
#     # print(args)
#
#     device = torch.device(args.device)
#
#     # fix the seed for reproducibility
#     seed = args.seed + utils.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True
#
#     model = get_model(args)
#     patch_size = model.patch_embed.patch_size
#     print("Patch size = %s" % str(patch_size))
#     args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
#     args.patch_size = patch_size
#
#     # get dataset
#     dataset_train = build_pretraining_dataset(args)
#     dataset_val = build_fixed_validation_dataset(args)
#     num_training_steps_per_epoch = len(dataset_train) // max(1, args.batch_size) // max(1, utils.get_world_size())
#
#     # dataset_val = build_validation_dataset(args)  # validate without fixed hint set
#
#     if args.distributed:
#         num_tasks = utils.get_world_size()
#         global_rank = utils.get_rank()
#         sampler_rank = global_rank
#         num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
#
#         sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
#         sampler_val = DistributedSampler(dataset_val, num_replicas=num_tasks, rank=sampler_rank, shuffle=False)
#         print("Sampler_train = %s" % str(sampler_train))
#     else:
#         sampler_train = RandomSampler(dataset_train)
#         sampler_val = RandomSampler(dataset_val)
#
#     if utils.get_rank() == 0 and args.log_dir is not None:
#         log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
#     else:
#         log_writer = None
#
#     data_loader_train = DataLoader(
#         dataset_train, sampler=sampler_train,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=True,
#         worker_init_fn=utils.seed_worker
#     )
#     data_loader_val = DataLoader(
#         dataset_val, sampler=sampler_val,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=False,
#         worker_init_fn=utils.seed_worker
#     )
#
#     model.to(device)
#     model_without_ddp = model
#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     print("Model = %s" % str(model_without_ddp))
#     print('number of params: {} M'.format(n_parameters / 1e6))
#
#     total_batch_size = args.batch_size * utils.get_world_size()
#     # args.lr = args.lr * total_batch_size / 256
#
#     print("LR = %.8f" % args.lr)
#     print("Batch size = %d" % total_batch_size)
#     print("Number of training steps = %d" % num_training_steps_per_epoch)
#     print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))
#
#     if args.distributed:
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
#         # For debugging
#         # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
#         model_without_ddp = model.module
#
#     optimizer = create_optimizer(args, model_without_ddp)
#     loss_scaler = NativeScaler()
#
#     print("Use step level LR & WD scheduler!")
#     lr_schedule_values = utils.cosine_scheduler(
#         args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
#         warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
#     if args.weight_decay_end is None:
#         args.weight_decay_end = args.weight_decay
#     wd_schedule_values = utils.cosine_scheduler(
#         args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
#     print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
#
#     utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp,
#                           optimizer=optimizer, loss_scaler=loss_scaler)
#     utils.save_args(args, args.output_dir, save_pkl=args.save_args_pkl, save_txt=args.save_args_txt)
#
#     print(f"Start training for {args.epochs} epochs")
#     start_time = time.time()
#     for epoch in range(args.start_epoch, args.epochs):
#         if args.distributed:
#             data_loader_train.sampler.set_epoch(epoch)
#         if log_writer is not None:
#             log_writer.set_step(epoch * num_training_steps_per_epoch)
#
#         train_stats = train_one_epoch(
#             model, data_loader_train,
#             optimizer, device, epoch, loss_scaler,
#             args.clip_grad, log_writer=log_writer,
#             start_steps=epoch * num_training_steps_per_epoch,
#             lr_schedule_values=lr_schedule_values,
#             wd_schedule_values=wd_schedule_values,
#             patch_size=patch_size[0],
#             exp_name=args.exp_name,
#         )
#         if args.output_dir:
#             if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
#                 val_stats = validate(
#                     model, data_loader_val, device, patch_size[0], log_writer,
#                     args.val_hint_list,
#                 )
#                 utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
#                                  optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
#
#         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}
#
#         print('epoch time {}'.format(str(datetime.timedelta(seconds=int(start_time)))))
#
#         if args.output_dir and utils.is_main_process():
#             if log_writer is not None:
#                 log_writer.flush()
#             with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
#                 f.write(json.dumps(log_stats) + "\n")
#
#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Training time {}'.format(total_time_str))
#
#
# if __name__ == '__main__':
#     import warnings
#     warnings.filterwarnings("ignore", category=UserWarning)
#     args = get_args()
#
#     # 1) compute exp name and dirs first
#     if not args.force_resume:
#         strtime = time.strftime("%y%m%d_%H%M%S")
#         args.exp_name = '_'.join([args.exp_name, strtime]) if args.exp_name else strtime
#         # set a default outputs root if user left it empty
#         if not args.output_dir:
#             args.output_dir = "./outputs"
#         if not args.log_dir:
#             args.log_dir = "tf_log"
#         args.output_dir = os.path.join(args.output_dir, args.model, args.exp_name)
#         args.log_dir = os.path.join(args.log_dir, args.exp_name)
#         os.makedirs(args.output_dir, exist_ok=True)
#         os.makedirs(args.log_dir, exist_ok=True)
#
#     # 2) set hint dirs
#     args.hint_dirs = [os.path.join(args.val_hint_dir, f'h{args.hint_size}-n{val_num_hint}')
#                       for val_num_hint in args.val_hint_list]
#
#     # 3) now it's safe to save args (utils.save_args is guaranteed to exist)
#     utils.save_args(args, args.output_dir, save_pkl=args.save_args_pkl, save_txt=args.save_args_txt)
#
#     # 4) run
#     main(args)
############################################################
################this is for thesis

#''''
# import argparse
# import datetime
# import json
# import os
# import random
# import time
# import pickle  # needed by _save_args_fallback
#
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# from timm.models import create_model
# from torch.optim import optimizer
# from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
#
# import modeling  # To register models
# import utils
# from datasets_org_till11thjune import build_pretraining_dataset, build_fixed_validation_dataset, build_validation_dataset
# from engine import train_one_epoch, validate
# from optim_factory import create_optimizer
# from utils import NativeScalerWithGradNormCount as NativeScaler
#
# import sys
#
# # --- Backfill utils.save_args if it's missing ---
# if not hasattr(utils, "save_args"):
#     def _save_args_wrapper(args, out_dir, save_pkl=True, save_txt=True):
#         # delegate to your local fallback
#         _save_args_fallback(args, out_dir, save_pkl=save_pkl, save_txt=save_txt)
#     utils.save_args = _save_args_wrapper
#
# # --- add to train.py ---
# def _save_args_fallback(args, out_dir, save_pkl=True, save_txt=True):
#     import json, os, pickle, datetime
#     os.makedirs(out_dir, exist_ok=True)
#     args_dict = vars(args)
#
#     # JSON
#     with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
#         json.dump(args_dict, f, indent=2, default=str)
#
#     # TXT (pretty)
#     if save_txt:
#         ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as f:
#             f.write(f"# iColoriT args saved at {ts}\n")
#             for k in sorted(args_dict.keys()):
#                 f.write(f"{k}: {args_dict[k]}\n")
#
#     # PKL (optional)
#     if save_pkl:
#         with open(os.path.join(out_dir, "args.pkl"), "wb") as f:
#             pickle.dump(args, f)
#
# class TeeLogger:
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#
#     def write(self, message):
#         self.terminal.write(message)  # Print to terminal
#         self.log.write(message)       # Write to log file
#         self.flush()
#
#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()
#
# # Redirect stdout and stderr
# #sys.stdout = TeeLogger(f"/media/sapanagupta/vol1/Sapana/data/temp/training_log_462_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
# #sys.stdout = TeeLogger(f"//home/sapanagupta/ICOLORIT_INPUTS/INPUTS/ORIGINAL_IMAGE/IcolorIt_Mask/results_masks/Cluster/Train/train_log{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
# sys.stdout = TeeLogger(f"/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/train_ablation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
# #/media/sapanagupta/vol1/Sapana/data/Threshold/training_log_462_threshold
# sys.stderr = sys.stdout
#
# def get_args():
#     parser = argparse.ArgumentParser('iColoriT training scripts', add_help=False)
#     # Training
#     parser.add_argument('--exp_name', default='', type=str)
#     parser.add_argument('--epochs', default=100, type=int)
#     parser.add_argument('--batch_size', default=32, type=int)
#     parser.add_argument('--num_workers', default=8, type=int)
#     parser.add_argument('--save_ckpt_freq', default=5, type=int)
#     parser.add_argument('--seed', default=4885, type=int)
#     parser.add_argument('--save_args_pkl', action='store_true', help='Save args as pickle file')
#     parser.add_argument('--no_save_args_pkl', action='store_false', dest='save_args_pkl', help='')
#     parser.set_defaults(save_args_pkl=True)
#     parser.add_argument('--save_args_txt', action='store_true', help='Save args as txt file')
#     # Dataset (NOTE: these names must match dataset builders)
#     parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
#     parser.add_argument('--data_path', default='data/train/images', type=str, help='dataset path (TRAIN images)')
#     parser.add_argument('--val_data_path', default='data/val/images', type=str, help='validation dataset path')
#     parser.add_argument('--val_hint_dir', type=str, default='data/hint', help='base dir for fixed validation hints (contains h2-n*)')
#     # Fixed-train-hint support (your patch-wise TXT hints)
#     parser.add_argument('--train_hint_base_dir', type=str, default=None, help='Base dir of TRAIN hints; contains h2-n* subfolders')
#     parser.add_argument('--train_num_hint', type=int, default=None, help='Number of hints to use for training; picks subdir h2-n{N}')
#     # near other data args
#     parser.add_argument(
#         '--hint_dirs',
#         type=str,
#         default='',
#         help='Comma-separated absolute paths to validation hint dirs (e.g. ".../h2-n0,.../h2-n20"). '
#              'If provided, overrides --val_hint_dir/--val_hint_list.'
#     )
#
#     # IO / resume
#     parser.add_argument('--output_dir', default='', help='path to save outputs')
#     parser.add_argument('--log_dir', default='tf_log', help='tensorboard log dir')
#     parser.add_argument('--resume', default='', help='checkpoint path')
#     parser.add_argument('--force_resume', action='store_true')
#     parser.add_argument('--start_epoch', default=0, type=int)
#     parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
#     parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
#     parser.set_defaults(pin_mem=True)
#     parser.add_argument('--gray_file_list_txt', type=str, default='', help='exclude these (optional)')
#     parser.add_argument('--return_name', action='store_true')
#     # Model
#     parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str)
#     parser.add_argument('--use_rpb', action='store_true')
#     parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
#     parser.set_defaults(use_rpb=True)
#     parser.add_argument('--head_mode', type=str, default='cnn', choices=['linear', 'cnn', 'locattn'])
#     parser.add_argument('--drop_path', type=float, default=0.0)
#     parser.add_argument('--mask_cent', action='store_true')
#     parser.add_argument('--avg_hint', action='store_true')
#     parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
#     parser.set_defaults(avg_hint=True)
#     # Hints
#     parser.add_argument('--hint_generator', type=str, default='RandomHintGenerator')
#     parser.add_argument('--num_hint_range', default=[0, 128], type=int, nargs=2)
#     parser.add_argument('--hint_size', default=4, type=int)
#     parser.add_argument('--val_hint_list', default=[1, 10, 20], nargs='+', type=int)
#     # LR / optim
#     parser.add_argument('--lr', type=float, default=5e-4)
#     parser.add_argument('--warmup_lr', type=float, default=1e-6)
#     parser.add_argument('--warmup_epochs', type=int, default=3)
#     parser.add_argument('--warmup_steps', type=int, default=-1)
#     parser.add_argument('--min_lr', type=float, default=1e-5)
#     parser.add_argument('--opt', default='adamw', type=str)
#     parser.add_argument('--opt_eps', default=1e-8, type=float)
#     parser.add_argument('--opt_betas', default=(0.9, 0.95), type=float, nargs='+')
#     parser.add_argument('--clip_grad', type=float, default=None)
#     parser.add_argument('--momentum', type=float, default=0.9)
#     parser.add_argument('--weight_decay', type=float, default=0.05)
#     parser.add_argument('--weight_decay_end', type=float, default=None)
#     # distributed
#     parser.add_argument('--device', default='cuda')
#     parser.add_argument('--world_size', default=1, type=int)
#     parser.add_argument('--local_rank', default=int(os.environ.get('LOCAL_RANK', 0)), type=int)
#     parser.add_argument('--dist_on_itp', action='store_true')
#     parser.add_argument('--dist_url', default='env://')
#     parser.add_argument('--auto_resume', action='store_true')
#     parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
#     parser.add_argument('--resume_weights_only', action='store_true', help='Only load model weights (skip optimizer)')
#     parser.set_defaults(auto_resume=True)
#     return parser.parse_args()
#
#
# # # # Use this to train the model from SCRATCH
# # def get_model(args):
# #     print(f"Creating model: {args.model}")
# #     model = create_model(
# #         args.model,
# #         pretrained=False,
# #         drop_path_rate=args.drop_path,
# #         drop_block_rate=None,
# #         use_rpb=args.use_rpb,
# #         avg_hint=args.avg_hint,
# #         head_mode=args.head_mode,
# #         mask_cent=args.mask_cent,
# #     )
# #     return model
#
# # Modified to run fine-tune model
# def get_model(args):
#     print(f"Creating model: {args.model}")
#     model = create_model(
#         args.model,
#         pretrained=True,  # Load pre-trained weights
#         drop_path_rate=args.drop_path,
#         drop_block_rate=None,
#         use_rpb=args.use_rpb,
#         avg_hint=args.avg_hint,
#         head_mode=args.head_mode,
#         mask_cent=args.mask_cent,
#     )
#
#     if args.resume:
#         checkpoint = torch.load(args.resume, map_location='cpu')
#         model.load_state_dict(checkpoint['model'], strict=False)
#         print(f"Loaded model weights from {args.resume}")
#
#         # Check if the flag --resume_weights_only is set
#         if not getattr(args, 'resume_weights_only', False):
#             # NOTE: we cannot load optimizer here because it doesn't exist yet in this scope.
#             # utils.auto_load_model below will properly restore optimizer/scaler.
#             print("Optimizer state will be restored later by utils.auto_load_model.")
#         else:
#             print("Skipping optimizer state loading (resume_weights_only flag is set).")
#
#     # Freeze all layers except the classification head (fine-tuning)
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # Unfreeze last transformer blocks and head
#     for name, param in model.named_parameters():
#         if 'blocks.10' in name or 'blocks.11' in name or 'head' in name:
#             param.requires_grad = True
#
#     print("Model fine-tuning enabled. All layers frozen except the head.")
#     return model
#
#
# # -------- SCHEDULE LENGTH CLAMP HELPER (fix for '303' assert) --------
# def _fit_schedule_len(schedule_list, epochs, niter_per_ep, name="lr"):
#     """
#     Make schedule length exactly epochs * niter_per_ep.
#     Truncates extra items or pads with the last value if short.
#     Accepts list/tuple/torch.Tensor. Returns a Python list.
#     """
#     if hasattr(schedule_list, "tolist"):
#         schedule = schedule_list.tolist()
#     else:
#         schedule = list(schedule_list)
#
#     target = int(epochs) * int(niter_per_ep)
#     if target <= 0:
#         raise ValueError(f"[{name} schedule] target length is non-positive "
#                          f"(epochs={epochs}, niter_per_ep={niter_per_ep}).")
#
#     if len(schedule) > target:
#         schedule = schedule[:target]
#     elif len(schedule) < target:
#         last = schedule[-1] if len(schedule) else 0.0
#         schedule.extend([last] * (target - len(schedule)))
#
#     return schedule
# # ---------------------------------------------------------------------
#
#
# def main(args):
#     utils.init_distributed_mode(args)
#     # print(args)
#
#     device = torch.device(args.device)
#
#     # fix the seed for reproducibility
#     seed = args.seed + utils.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True
#
#     model = get_model(args)
#     patch_size = model.patch_embed.patch_size
#     print("Patch size = %s" % str(patch_size))
#     args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
#     args.patch_size = patch_size
#
#     # get dataset
#     dataset_train = build_pretraining_dataset(args)
#     dataset_val = build_fixed_validation_dataset(args)
#     num_training_steps_per_epoch = len(dataset_train) // max(1, args.batch_size) // max(1, utils.get_world_size())
#
#     # dataset_val = build_validation_dataset(args)  # validate without fixed hint set
#
#     if args.distributed:
#         num_tasks = utils.get_world_size()
#         global_rank = utils.get_rank()
#         sampler_rank = global_rank
#         num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
#
#         sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
#         sampler_val = DistributedSampler(dataset_val, num_replicas=num_tasks, rank=sampler_rank, shuffle=False)
#         print("Sampler_train = %s" % str(sampler_train))
#     else:
#         sampler_train = RandomSampler(dataset_train)
#         sampler_val = RandomSampler(dataset_val)
#
#     if utils.get_rank() == 0 and args.log_dir is not None:
#         log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
#     else:
#         log_writer = None
#
#     data_loader_train = DataLoader(
#         dataset_train, sampler=sampler_train,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=True,
#         worker_init_fn=utils.seed_worker
#     )
#     data_loader_val = DataLoader(
#         dataset_val, sampler=sampler_val,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=False,
#         worker_init_fn=utils.seed_worker
#     )
#
#     model.to(device)
#     model_without_ddp = model
#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     print("Model = %s" % str(model_without_ddp))
#     print('number of params: {} M'.format(n_parameters / 1e6))
#
#     total_batch_size = args.batch_size * utils.get_world_size()
#     # args.lr = args.lr * total_batch_size / 256
#
#     print("LR = %.8f" % args.lr)
#     print("Batch size = %d" % total_batch_size)
#
#     # Use the ACTUAL iterations per epoch from the DataLoader (fix for schedule assert)
#     niter_per_ep = len(data_loader_train)
#     if niter_per_ep == 0:
#         raise RuntimeError("DataLoader has 0 iterations. Check dataset size / batch size / sampler.")
#     print("Number of training steps = %d" % niter_per_ep)
#     print("Number of training examples per epoch = %d" % (total_batch_size * niter_per_ep))
#
#     if args.distributed:
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
#         # For debugging
#         # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
#         model_without_ddp = model.module
#
#     optimizer = create_optimizer(args, model_without_ddp)
#     loss_scaler = NativeScaler()
#
#     # -------- Build LR/WD schedules using niter_per_ep and clamp lengths --------
#     print("Use step level LR & WD scheduler!")
#     print(f"[SCHED] epochs={args.epochs}, niter_per_ep={niter_per_ep}, "
#           f"total_updates={args.epochs * niter_per_ep}")
#
#     lr_schedule_values = utils.cosine_scheduler(
#         args.lr, args.min_lr, args.epochs, niter_per_ep,
#         warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps
#     )
#     if args.weight_decay_end is None:
#         args.weight_decay_end = args.weight_decay
#     wd_schedule_values = utils.cosine_scheduler(
#         args.weight_decay, args.weight_decay_end, args.epochs, niter_per_ep
#     )
#
#     # Clamp to exactly epochs * niter_per_ep (prevents assertion mismatch)
#     lr_schedule_values = _fit_schedule_len(lr_schedule_values, args.epochs, niter_per_ep, name="lr")
#     wd_schedule_values = _fit_schedule_len(wd_schedule_values, args.epochs, niter_per_ep, name="wd")
#
#     print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
#     print(f"[SCHED] lr_len={len(lr_schedule_values)}, wd_len={len(wd_schedule_values)}")
#     # ---------------------------------------------------------------------------
#
#     utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp,
#                           optimizer=optimizer, loss_scaler=loss_scaler)
#     utils.save_args(args, args.output_dir, save_pkl=args.save_args_pkl, save_txt=args.save_args_txt)
#
#     print(f"Start training for {args.epochs} epochs")
#     start_time = time.time()
#     for epoch in range(args.start_epoch, args.epochs):
#         if args.distributed:
#             data_loader_train.sampler.set_epoch(epoch)
#         if log_writer is not None:
#             log_writer.set_step(epoch * niter_per_ep)
#
#         train_stats = train_one_epoch(
#             model, data_loader_train,
#             optimizer, device, epoch, loss_scaler,
#             args.clip_grad, log_writer=log_writer,
#             start_steps=epoch * niter_per_ep,
#             lr_schedule_values=lr_schedule_values,
#             wd_schedule_values=wd_schedule_values,
#             patch_size=patch_size[0],
#             exp_name=args.exp_name,
#         )
#         if args.output_dir:
#             if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
#                 val_stats = validate(
#                     model, data_loader_val, device, patch_size[0], log_writer,
#                     args.val_hint_list,
#                 )
#                 utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
#                                  optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
#
#         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}
#
#         print('epoch time {}'.format(str(datetime.timedelta(seconds=int(start_time)))))
#
#         if args.output_dir and utils.is_main_process():
#             if log_writer is not None:
#                 log_writer.flush()
#             with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
#                 f.write(json.dumps(log_stats) + "\n")
#
#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Training time {}'.format(total_time_str))
#
#
# if __name__ == '__main__':
#     import warnings
#     warnings.filterwarnings("ignore", category=UserWarning)
#     args = get_args()
#
#     # 1) compute exp name and dirs first
#     if not args.force_resume:
#         strtime = time.strftime("%y%m%d_%H%M%S")
#         args.exp_name = '_'.join([args.exp_name, strtime]) if args.exp_name else strtime
#         # set a default outputs root if user left it empty
#         if not args.output_dir:
#             args.output_dir = "./outputs"
#         if not args.log_dir:
#             args.log_dir = "tf_log"
#         args.output_dir = os.path.join(args.output_dir, args.model, args.exp_name)
#         args.log_dir = os.path.join(args.log_dir, args.exp_name)
#         os.makedirs(args.output_dir, exist_ok=True)
#         os.makedirs(args.log_dir, exist_ok=True)
#
#     # 2) set hint dirs
#     args.hint_dirs = [os.path.join(args.val_hint_dir, f'h{args.hint_size}-n{val_num_hint}')
#                       for val_num_hint in args.val_hint_list]
#
#     # 3) now it's safe to save args (utils.save_args is guaranteed to exist)
#     utils.save_args(args, args.output_dir, save_pkl=args.save_args_pkl, save_txt=args.save_args_txt)
#
#     # 4) run
#     main(args)


#####the above is correct but for valid only, i have just pointed to it valid path

##########################################

import argparse
import datetime
import json
import os
import random
import time
import pickle  # needed by _save_args_fallback
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

import modeling  # To register models
import utils
from datasets_org_till11thjune import (
    build_pretraining_dataset,
    build_fixed_validation_dataset,
    build_validation_dataset,
)
from engine import train_one_epoch, validate
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler


# --- Backfill utils.save_args if it's missing ---
if not hasattr(utils, "save_args"):
    def _save_args_wrapper(args, out_dir, save_pkl=True, save_txt=True):
        _save_args_fallback(args, out_dir, save_pkl=save_pkl, save_txt=save_txt)
    utils.save_args = _save_args_wrapper


def _save_args_fallback(args, out_dir, save_pkl=True, save_txt=True):
    os.makedirs(out_dir, exist_ok=True)
    args_dict = vars(args)

    # JSON
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, default=str)

    # TXT (pretty)
    if save_txt:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as f:
            f.write(f"# iColoriT args saved at {ts}\n")
            for k in sorted(args_dict.keys()):
                f.write(f"{k}: {args_dict[k]}\n")

    # PKL (optional)
    if save_pkl:
        with open(os.path.join(out_dir, "args.pkl"), "wb") as f:
            pickle.dump(args, f)


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Redirect stdout and stderr to a rolling file (keeps console prints too)
sys.stdout = TeeLogger(
    f"/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/train_ablation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)
sys.stderr = sys.stdout


def get_args():
    parser = argparse.ArgumentParser('iColoriT training scripts', add_help=False)

    # Training
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--save_args_pkl', action='store_true', help='Save args as pickle file')
    parser.add_argument('--no_save_args_pkl', action='store_false', dest='save_args_pkl')
    parser.set_defaults(save_args_pkl=True)
    parser.add_argument('--save_args_txt', action='store_true', help='Save args as txt file')

    # Dataset (NOTE: these names must match dataset builders)
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--data_path', default='data/train/images', type=str, help='dataset path (TRAIN images)')
    parser.add_argument('--val_data_path', default='data/val/images', type=str, help='validation dataset path')
    parser.add_argument('--val_hint_dir', type=str, default='data/hint', help='base dir for fixed validation hints (contains h2-n*)')

    # Fixed-train-hint support (patch-wise TXT hints)
    parser.add_argument('--train_hint_base_dir', type=str, default=None, help='Base dir of TRAIN hints; contains h2-n* subfolders')
    parser.add_argument('--train_num_hint', type=int, default=None, help='Number of hints to use for training; picks subdir h2-n{N}')

    # Optional direct list of validation hint directories (comma-separated)
    parser.add_argument(
        '--hint_dirs',
        type=str,
        default='',
        help='Comma-separated absolute paths to validation hint dirs (e.g. ".../h2-n0,.../h2-n20"). '
             'If provided, overrides --val_hint_dir/--val_hint_list.'
    )

    # IO / resume
    parser.add_argument('--output_dir', default='', help='path to save outputs')
    parser.add_argument('--log_dir', default='tf_log', help='tensorboard log dir')
    parser.add_argument('--resume', default='', help='checkpoint path')
    parser.add_argument('--force_resume', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--gray_file_list_txt', type=str, default='', help='exclude these (optional)')
    parser.add_argument('--return_name', action='store_true')

    # Model
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str)
    parser.add_argument('--use_rpb', action='store_true')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--head_mode', type=str, default='cnn', choices=['linear', 'cnn', 'locattn'])
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--mask_cent', action='store_true')
    parser.add_argument('--avg_hint', action='store_true')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)

    # Hints
    parser.add_argument('--hint_generator', type=str, default='RandomHintGenerator')
    parser.add_argument('--num_hint_range', default=[0, 128], type=int, nargs=2)
    parser.add_argument('--hint_size', default=4, type=int)
    parser.add_argument('--val_hint_list', default=[1, 10, 20], nargs='+', type=int)

    # LR / optim
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=(0.9, 0.95), type=float, nargs='+')
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)

    # distributed
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=int(os.environ.get('LOCAL_RANK', 0)), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.add_argument('--resume_weights_only', action='store_true', help='Only load model weights (skip optimizer)')
    parser.set_defaults(auto_resume=True)

    return parser.parse_args()


# -------- MODEL --------
def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,  # Load pre-trained weights
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
        mask_cent=args.mask_cent,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded model weights from {args.resume}")
        if not getattr(args, 'resume_weights_only', False):
            print("Optimizer state will be restored later by utils.auto_load_model.")
        else:
            print("Skipping optimizer state loading (resume_weights_only flag is set).")

    # Fine-tune last blocks + head
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if 'blocks.10' in name or 'blocks.11' in name or 'head' in name:
            p.requires_grad = True
    print("Model fine-tuning enabled. All layers frozen except the head.")
    return model


# -------- SCHEDULE LENGTH CLAMP --------
def _fit_schedule_len(schedule_list, epochs, niter_per_ep, name="lr"):
    if hasattr(schedule_list, "tolist"):
        schedule = schedule_list.tolist()
    else:
        schedule = list(schedule_list)

    target = int(epochs) * int(niter_per_ep)
    if target <= 0:
        raise ValueError(f"[{name} schedule] target length is non-positive "
                         f"(epochs={epochs}, niter_per_ep={niter_per_ep}).")

    if len(schedule) > target:
        schedule = schedule[:target]
    elif len(schedule) < target:
        last = schedule[-1] if len(schedule) else 0.0
        schedule.extend([last] * (target - len(schedule)))

    return schedule


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # seeds
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # Datasets
    dataset_train = build_pretraining_dataset(args)
    dataset_val = build_fixed_validation_dataset(args)
    num_training_steps_per_epoch = len(dataset_train) // max(1, args.batch_size) // max(1, utils.get_world_size())

    if args.distributed:
        num_tasks = utils.get_world_size()
        sampler_rank = utils.get_rank()
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
        sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, num_replicas=num_tasks, rank=sampler_rank, shuffle=False)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = RandomSampler(dataset_val)

    log_writer = utils.TensorboardLogger(log_dir=args.log_dir) if (utils.get_rank() == 0 and args.log_dir) else None

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=utils.seed_worker
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)

    niter_per_ep = len(data_loader_train)
    if niter_per_ep == 0:
        raise RuntimeError("DataLoader has 0 iterations. Check dataset size / batch size / sampler.")
    print("Number of training steps = %d" % niter_per_ep)
    print("Number of training examples per epoch = %d" % (total_batch_size * niter_per_ep))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    # -------- Schedulers --------
    print("Use step level LR & WD scheduler!")
    print(f"[SCHED] epochs={args.epochs}, niter_per_ep={niter_per_ep}, total_updates={args.epochs * niter_per_ep}")

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, niter_per_ep,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, niter_per_ep
    )
    lr_schedule_values = _fit_schedule_len(lr_schedule_values, args.epochs, niter_per_ep, name="lr")
    wd_schedule_values = _fit_schedule_len(wd_schedule_values, args.epochs, niter_per_ep, name="wd")
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    print(f"[SCHED] lr_len={len(lr_schedule_values)}, wd_len={len(wd_schedule_values)}")

    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp,
                          optimizer=optimizer, loss_scaler=loss_scaler)
    utils.save_args(args, args.output_dir, save_pkl=args.save_args_pkl, save_txt=args.save_args_txt)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * niter_per_ep)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * niter_per_ep,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            exp_name=args.exp_name,
        )

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                val_stats = validate(
                    model, data_loader_val, device, patch_size[0], log_writer,
                    args.val_hint_list,
                )
                utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                 optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}
        print('epoch time {}'.format(str(datetime.timedelta(seconds=int(start_time)))))

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()

    # 1) Compute experiment/output dirs first
    if not args.force_resume:
        strtime = time.strftime("%y%m%d_%H%M%S")
        args.exp_name = '_'.join([args.exp_name, strtime]) if args.exp_name else strtime
        if not args.output_dir:
            args.output_dir = "./outputs"
        if not args.log_dir:
            args.log_dir = "tf_log"
        args.output_dir = os.path.join(args.output_dir, args.model, args.exp_name)
        args.log_dir = os.path.join(args.log_dir, args.exp_name)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    # 2) Normalize validation hint dirs — supports either --hint_dirs or base+list
    norm_hint_dirs = []
    if isinstance(args.hint_dirs, str) and args.hint_dirs.strip():
        norm_hint_dirs = [p.strip() for p in args.hint_dirs.split(',') if p.strip()]
    else:
        base = getattr(args, 'val_hint_dir', '')
        levels = getattr(args, 'val_hint_list', [])
        if base and levels:
            norm_hint_dirs = [os.path.join(base, f"h{args.hint_size}-n{int(n)}") for n in levels]

    # Keep only existing dirs (avoid silent mismatch)
    existing = [d for d in norm_hint_dirs if os.path.isdir(d)]
    missing = [d for d in norm_hint_dirs if not os.path.isdir(d)]
    args.hint_dirs = existing

    # 3) Friendly summary (helps catch mismatches fast)
    print("==============================================")
    print("Starting iColoriT Fine-Tuning")
    print(f"Training Patches Path: {args.data_path}")
    print(f"Validation Images Path: {args.val_data_path}")
    print(f"Validation Hint Dirs:   {','.join(args.hint_dirs) if args.hint_dirs else '(none)'}")
    if missing:
        print(f"[WARN] These hint dirs do not exist and were ignored: {', '.join(missing)}")
    print(f"Output will be saved to: {args.output_dir}")
    print("==============================================")

    # 4) Run
    main(args)











