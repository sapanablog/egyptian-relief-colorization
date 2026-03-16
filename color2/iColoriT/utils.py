# ###############################################################################################
# ###############################################################################################
# # --------------------------------------------------------
# # --------------------------------------------------------
# # Based on BEiT, timm, DINO and DeiT code bases
# # https://github.com/microsoft/unilm/tree/master/beit
# # https://github.com/rwightman/pytorch-image-models/tree/master/timm
# # https://github.com/facebookresearch/deit
# # https://github.com/facebookresearch/dino
# # --------------------------------------------------------'
# import datetime
# import io
# import json
# import math
# import os
# import pickle
# import random
# import time
# from collections import defaultdict, deque
# from pathlib import Path
#
# import numpy as np
# import torch
# import torch.distributed as dist
# from tensorboardX import SummaryWriter
# from timm.utils import get_state_dict
#
# #####I am going to replace this in below line import math
# inf = math.inf
# #from torch._six import inf
#
# ####################################### Utils #######################################
#
#
# def get_args_table(args_dict):
#     return get_pretty_table(args_dict, field_names=['Arg', 'Value'])
#
#
# def get_pretty_table(saved_dict, field_names=None, **kwargs):
#     from prettytable import PrettyTable
#     table = PrettyTable(field_names, **kwargs)
#     for arg, val in saved_dict.items():
#         if isinstance(val, (list, tuple)) and len(field_names) == len(val) + 1:
#             table.add_row([arg, *val])
#         else:
#             table.add_row([arg, val])
#     return table
#
#
# def save_args(args, save_dir, name='args', save_pkl=True, save_txt=False):
#     assert save_pkl or save_txt
#     os.makedirs(save_dir, exist_ok=True)
#     if save_pkl:
#         # Save args
#         with open(os.path.join(save_dir, f'{name}.pkl'), "wb") as f:
#             pickle.dump(args, f)
#         print(f'{name} is saved on {os.path.join(save_dir, f"{name}.pkl")}')
#     if save_txt:
#         # Save args table
#         args_table = get_args_table(vars(args) if not isinstance(args, dict) else args)
#         with open(os.path.join(save_dir, f'{name}_table.txt'), "w") as f:
#             f.write(str(args_table))
#         print(f'{name} is saved on {os.path.join(save_dir, f"{name}_table.txt")}')
#
#
# ######################################## Logger #########################################
#
#
# class SmoothedValue(object):
#     """Track a series of values and provide access to smoothed values over a
#     window or the global series average.
#     """
#
#     def __init__(self, window_size=20, fmt=None):
#         if fmt is None:
#             fmt = "{median:.4f} ({global_avg:.4f})"
#         self.deque = deque(maxlen=window_size)
#         self.total = 0.0
#         self.count = 0
#         self.fmt = fmt
#
#     def update(self, value, n=1):
#         self.deque.append(value)
#         self.count += n
#         self.total += value * n
#
#     def synchronize_between_processes(self):
#         """
#         Warning: does not synchronize the deque!
#         """
#         if not is_dist_avail_and_initialized():
#             return
#         t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
#         dist.barrier()
#         dist.all_reduce(t)
#         t = t.tolist()
#         self.count = int(t[0])
#         self.total = t[1]
#
#     @property
#     def median(self):
#         d = torch.tensor(list(self.deque))
#         return d.median().item()
#
#     @property
#     def avg(self):
#         d = torch.tensor(list(self.deque), dtype=torch.float32)
#         return d.mean().item()
#
#     @property
#     def global_avg(self):
#         return self.total / self.count
#
#     # ##this is modified by me on 27thjune
#     # @property
#     # def global_avg(self):
#     #     # --- SAFER IMPLEMENTATION (prevents ZeroDivisionError) ---
#     #     if self.count == 0:          # happens before the first update
#     #         return 0.0
#     #     return self.total / self.count
#
#     @property
#     def max(self):
#         return max(self.deque)
#
#     @property
#     def value(self):
#         return self.deque[-1]
#
#     def __str__(self):
#         return self.fmt.format(
#             median=self.median,
#             avg=self.avg,
#             global_avg=self.global_avg,
#             max=self.max,
#             value=self.value)
#
#
# class MetricLogger(object):
#     def __init__(self, delimiter="\t"):
#         self.meters = defaultdict(SmoothedValue)
#         self.delimiter = delimiter
#
#     def update(self, **kwargs):
#         for k, v in kwargs.items():
#             if v is None:
#                 continue
#             if isinstance(v, torch.Tensor):
#                 v = v.item()
#             assert isinstance(v, (float, int))
#             self.meters[k].update(v)
#
#     def __getattr__(self, attr):
#         if attr in self.meters:
#             return self.meters[attr]
#         if attr in self.__dict__:
#             return self.__dict__[attr]
#         raise AttributeError("'{}' object has no attribute '{}'".format(
#             type(self).__name__, attr))
#
#     def __str__(self):
#         loss_str = []
#         for name, meter in self.meters.items():
#             loss_str.append(
#                 "{}: {}".format(name, str(meter))
#             )
#         return self.delimiter.join(loss_str)
#
#     def synchronize_between_processes(self):
#         for meter in self.meters.values():
#             meter.synchronize_between_processes()
#
#     def add_meter(self, name, meter):
#         self.meters[name] = meter
#
#     def log_every(self, iterable, print_freq, header=None):
#         i = 0
#         if not header:
#             header = ''
#         start_time = time.time()
#         end = time.time()
#         iter_time = SmoothedValue(fmt='{avg:.4f}')
#         data_time = SmoothedValue(fmt='{avg:.4f}')
#         space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
#         log_msg = [
#             header,
#             '[{0' + space_fmt + '}/{1}]',
#             'eta: {eta}',
#             '{meters}',
#             'time: {time}',
#             'data: {data}'
#         ]
#         if torch.cuda.is_available():
#             log_msg.append('max mem: {memory:.0f}')
#         log_msg = self.delimiter.join(log_msg)
#         MB = 1024.0 * 1024.0
#         for obj in iterable:
#             data_time.update(time.time() - end)
#             yield obj
#             iter_time.update(time.time() - end)
#             if i % print_freq == 0 or i == len(iterable) - 1:
#                 eta_seconds = iter_time.global_avg * (len(iterable) - i)
#                 eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#                 if torch.cuda.is_available():
#                     print(log_msg.format(
#                         i, len(iterable), eta=eta_string,
#                         meters=str(self),
#                         time=str(iter_time), data=str(data_time),
#                         memory=torch.cuda.max_memory_allocated() / MB))
#                 else:
#                     print(log_msg.format(
#                         i, len(iterable), eta=eta_string,
#                         meters=str(self),
#                         time=str(iter_time), data=str(data_time)))
#             i += 1
#             end = time.time()
#         total_time = time.time() - start_time
#         total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#         print('{} Total time: {} ({:.4f} s / it)'.format(
#             header, total_time_str, total_time / len(iterable)))
#
#
# class TensorboardLogger(object):
#     def __init__(self, log_dir):
#         self.writer = SummaryWriter(logdir=log_dir)
#         self.step = 0
#
#     def set_step(self, step=None):
#         if step is not None:
#             self.step = step
#         else:
#             self.step += 1
#
#     def update(self, head='scalar', step=None, **kwargs):
#         for k, v in kwargs.items():
#             if v is None:
#                 continue
#             if isinstance(v, torch.Tensor):
#                 v = v.item()
#             assert isinstance(v, (float, int))
#             self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)
#
#     def flush(self):
#         self.writer.flush()
#
# ########################################### Distributed Training #######################################
#
#
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
#
#
# def _load_checkpoint_for_ema(model_ema, checkpoint):
#     """
#     Workaround for ModelEma._load_checkpoint to accept an already-loaded object
#     """
#     mem_file = io.BytesIO()
#     torch.save(checkpoint, mem_file)
#     mem_file.seek(0)
#     model_ema._load_checkpoint(mem_file)
#
#
# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__
#     builtin_print = __builtin__.print
#
#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)
#
#     __builtin__.print = print
#
#
# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True
#
#
# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()
#
#
# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()
#
#
# def is_main_process():
#     return get_rank() == 0
#
#
# def save_on_master(*args, **kwargs):
#     if is_main_process():
#         torch.save(*args, **kwargs)
#
#
# def init_distributed_mode(args):
#     if args.dist_on_itp:
#         args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
#         args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
#         args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
#         args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
#         os.environ['LOCAL_RANK'] = str(args.gpu)
#         os.environ['RANK'] = str(args.rank)
#         os.environ['WORLD_SIZE'] = str(args.world_size)
#         # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
#     elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = args.rank % torch.cuda.device_count()
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return
#
#     args.distributed = True
#
#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}, gpu {}'.format(
#         args.rank, args.dist_url, args.gpu), flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank)
#     torch.distributed.barrier()
#     setup_for_distributed(args.rank == 0)
#
#
# def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
#     missing_keys = []
#     unexpected_keys = []
#     error_msgs = []
#     # copy state_dict so _load_from_state_dict can modify it
#     metadata = getattr(state_dict, '_metadata', None)
#     state_dict = state_dict.copy()
#     if metadata is not None:
#         state_dict._metadata = metadata
#
#     def load(module, prefix=''):
#         local_metadata = {} if metadata is None else metadata.get(
#             prefix[:-1], {})
#         module._load_from_state_dict(
#             state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
#         for name, child in module._modules.items():
#             if child is not None:
#                 load(child, prefix + name + '.')
#
#     load(model, prefix=prefix)
#
#     warn_missing_keys = []
#     ignore_missing_keys = []
#     for key in missing_keys:
#         keep_flag = True
#         for ignore_key in ignore_missing.split('|'):
#             if ignore_key in key:
#                 keep_flag = False
#                 break
#         if keep_flag:
#             warn_missing_keys.append(key)
#         else:
#             ignore_missing_keys.append(key)
#
#     missing_keys = warn_missing_keys
#
#     if len(missing_keys) > 0:
#         print("Weights of {} not initialized from pretrained model: {}".format(
#             model.__class__.__name__, missing_keys))
#     if len(unexpected_keys) > 0:
#         print("Weights from pretrained model not used in {}: {}".format(
#             model.__class__.__name__, unexpected_keys))
#     if len(ignore_missing_keys) > 0:
#         print("Ignored weights of {} not initialized from pretrained model: {}".format(
#             model.__class__.__name__, ignore_missing_keys))
#     if len(error_msgs) > 0:
#         print('\n'.join(error_msgs))
#
#
# class NativeScalerWithGradNormCount:
#     state_dict_key = "amp_scaler"
#
#     def __init__(self):
#         self._scaler = torch.cuda.amp.GradScaler()
#
#     def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
#         self._scaler.scale(loss).backward(create_graph=create_graph)
#         if update_grad:
#             if clip_grad is not None:
#                 assert parameters is not None
#                 self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
#                 norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
#             else:
#                 self._scaler.unscale_(optimizer)
#                 norm = get_grad_norm_(parameters)
#             self._scaler.step(optimizer)
#             self._scaler.update()
#         else:
#             norm = None
#         return norm
#
#     def state_dict(self):
#         return self._scaler.state_dict()
#
#     def load_state_dict(self, state_dict):
#         self._scaler.load_state_dict(state_dict)
#
#
# def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     parameters = [p for p in parameters if p.grad is not None]
#     norm_type = float(norm_type)
#     if len(parameters) == 0:
#         return torch.tensor(0.)
#     device = parameters[0].grad.device
#     if norm_type == inf:
#         total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
#     else:
#         total_norm = torch.norm(torch.stack(
#             [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
#     return total_norm
#
#
# def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
#                      start_warmup_value=0, warmup_steps=-1):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_steps > 0:
#         warmup_iters = warmup_steps
#     print("Set warmup steps = %d" % warmup_iters)
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
#
#     iters = np.arange(epochs * niter_per_ep - warmup_iters)
#     schedule = np.array(
#         [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
#
#     schedule = np.concatenate((warmup_schedule, schedule))
#
#     assert len(schedule) == epochs * niter_per_ep
#     return schedule
#
#
# def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
#     output_dir = Path(args.output_dir)
#     epoch_name = str(epoch)
#     if loss_scaler is not None:
#         checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
#         for checkpoint_path in checkpoint_paths:
#             to_save = {
#                 'model': model_without_ddp.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'scaler': loss_scaler.state_dict(),
#                 'args': args,
#             }
#
#             if model_ema is not None:
#                 to_save['model_ema'] = get_state_dict(model_ema)
#
#             save_on_master(to_save, checkpoint_path)
#     else:
#         client_state = {'epoch': epoch}
#         if model_ema is not None:
#             client_state['model_ema'] = get_state_dict(model_ema)
#         model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)
#
#
# # def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
# #     output_dir = Path(args.output_dir)
# #     if loss_scaler is not None:
# #         # torch.amp
# #         if args.auto_resume and len(args.resume) == 0:
# #             import glob
# #             all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
# #             latest_ckpt = -1
# #             for ckpt in all_checkpoints:
# #                 t = ckpt.split('-')[-1].split('.')[0]
# #                 if t.isdigit():
# #                     latest_ckpt = max(int(t), latest_ckpt)
# #             if latest_ckpt >= 0:
# #                 args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
# #             print("Auto resume checkpoint: %s" % args.resume)
# #
# #         if args.resume:
# #             if args.resume.startswith('https'):
# #                 checkpoint = torch.hub.load_state_dict_from_url(
# #                     args.resume, map_location='cpu', check_hash=True)
# #             else:
# #                 checkpoint = torch.load(args.resume, map_location='cpu')
# #             model_without_ddp.load_state_dict(checkpoint['model'])
# #             print("Resume checkpoint %s" % args.resume)
# #             if 'optimizer' in checkpoint and 'epoch' in checkpoint:
# #                 optimizer.load_state_dict(checkpoint['optimizer'])
# #                 args.start_epoch = checkpoint['epoch'] + 1
# #                 if hasattr(args, 'model_ema') and args.model_ema:
# #                     _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
# #                 if 'scaler' in checkpoint:
# #                     loss_scaler.load_state_dict(checkpoint['scaler'])
# #                 print("With optim & sched!")
# #     else:
# #         # deepspeed, only support '--auto_resume'.
# #         if args.auto_resume:
# #             import glob
# #             all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
# #             latest_ckpt = -1
# #             for ckpt in all_checkpoints:
# #                 t = ckpt.split('-')[-1].split('.')[0]
# #                 if t.isdigit():
# #                     latest_ckpt = max(int(t), latest_ckpt)
# #             if latest_ckpt >= 0:
# #                 args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
# #                 print("Auto resume checkpoint: %d" % latest_ckpt)
# #                 _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
# #                 args.start_epoch = client_states['epoch'] + 1
# #                 if model_ema is not None:
# #                     if args.model_ema:
# #                         _load_checkpoint_for_ema(model_ema, client_states['model_ema'])
# #
#
# # Only uncomment when you want to use fine-tune model, for scratch use above
# def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler):
#     checkpoint = torch.load(args.resume, map_location='cpu')
#     print(f"Resume checkpoint {args.resume}")
#
#     # Always load model weights
#     model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
#
#     # Skip optimizer loading if resume_weights_only is set
#     if not getattr(args, 'resume_weights_only', False):
#         if 'optimizer' in checkpoint:
#             try:
#                 optimizer.load_state_dict(checkpoint['optimizer'])
#                 print("Loaded optimizer state from checkpoint.")
#             except ValueError as e:
#                 print(f"Warning: Optimizer state mismatch. Skipping optimizer loading. Error: {e}")
#         else:
#             print("No optimizer state found in checkpoint.")
#     else:
#         print("Skipping optimizer state loading (resume_weights_only flag is set).")
#
#     # Load scaler if it exists
#     if 'scaler' in checkpoint and loss_scaler is not None:
#         loss_scaler.load_state_dict(checkpoint['scaler'])
#
#
# def create_ds_config(args):
#     args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
#     with open(args.deepspeed_config, mode="w") as writer:
#         ds_config = {
#             "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
#             "train_micro_batch_size_per_gpu": args.batch_size,
#             "steps_per_print": 1000,
#             "optimizer": {
#                 "type": "Adam",
#                 "adam_w_mode": True,
#                 "params": {
#                     "lr": args.lr,
#                     "weight_decay": args.weight_decay,
#                     "bias_correction": True,
#                     "betas": [
#                         0.9,
#                         0.999
#                     ],
#                     "eps": 1e-8
#                 }
#             },
#             "fp16": {
#                 "enabled": True,
#                 "loss_scale": 0,
#                 "initial_scale_power": 7,
#                 "loss_scale_window": 128
#             }
#         }
#
#         writer.write(json.dumps(ds_config, indent=2))
#
#
# ############################################### Color Conversion #################################################
#
# def rgb2xyz(rgb):  # rgb from [0,1]
#     # array([[0.412453, 0.357580, 0.180423],
#     #        [0.212671, 0.715160, 0.072169],
#     #        [0.019334, 0.119193, 0.950227]])
#
#     mask = (rgb > .04045).type(torch.FloatTensor)
#     if(rgb.is_cuda):
#         mask = mask.cuda()
#
#     rgb = (((rgb + .055) / 1.055)**2.4) * mask + rgb / 12.92 * (1 - mask)
#
#     print('rgb.........:', rgb, rgb.shape)
#
#     x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
#     y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
#     z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
#     out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)
#
#     return out
#
#
# def xyz2rgb(xyz):
#     # array([[ 3.24048134, -1.53715152, -0.49853633],
#     #        [-0.96925495,  1.87599   ,  0.04155593],
#     #        [ 0.05564664, -0.20404134,  1.05731107]])
#
#     r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
#     g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
#     b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]
#
#     rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
#     rgb = torch.max(rgb, torch.zeros_like(rgb))  # sometimes reaches a small negative number, which causes NaNs
#
#     mask = (rgb > .0031308).type(torch.FloatTensor)
#     if(rgb.is_cuda):
#         mask = mask.cuda()
#
#     rgb = (1.055 * (rgb**(1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)
#
#     return rgb
#
#
# def xyz2lab(xyz):
#     # 0.95047, 1., 1.08883 # white
#     sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
#     if(xyz.is_cuda):
#         sc = sc.cuda()
#
#     xyz_scale = xyz / sc
#
#     mask = (xyz_scale > .008856).type(torch.FloatTensor)
#     if(xyz_scale.is_cuda):
#         mask = mask.cuda()
#
#     xyz_int = xyz_scale**(1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)
#
#     L = 116. * xyz_int[:, 1, :, :] - 16.
#     a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
#     b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
#     out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)
#
#     return out
#
#
# def lab2xyz(lab):
#     y_int = (lab[:, 0, :, :] + 16.) / 116.
#     x_int = (lab[:, 1, :, :] / 500.) + y_int
#     z_int = y_int - (lab[:, 2, :, :] / 200.)
#     if(z_int.is_cuda):
#         z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
#     else:
#         z_int = torch.max(torch.Tensor((0,)), z_int)
#
#     out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
#     mask = (out > .2068966).type(torch.FloatTensor)
#     if(out.is_cuda):
#         mask = mask.cuda()
#
#     out = (out**3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)
#
#     sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
#     sc = sc.to(out.device)
#
#     out = out * sc
#
#     return out
#
#
# def rgb2lab(rgb, l_cent=50, l_norm=100, ab_norm=110):
#     lab = xyz2lab(rgb2xyz(rgb))
#     l_rs = (lab[:, [0], :, :] - l_cent) / l_norm
#     ab_rs = lab[:, 1:, :, :] / ab_norm
#     out = torch.cat((l_rs, ab_rs), dim=1)
#
#     return out
#
#
# def lab2rgb(lab_rs, l_cent=50, l_norm=100, ab_norm=110):
#     l = lab_rs[:, [0], :, :] * l_norm + l_cent
#     ab = lab_rs[:, 1:, :, :] * ab_norm
#     lab = torch.cat((l, ab), dim=1)
#     out = xyz2rgb(lab2xyz(lab))
#
#     return out
#
#
# ############################################## Evaluation #############################################
#
# def psnr(img1: torch.Tensor, img2: torch.Tensor, epsilon=1e-5) -> torch.Tensor:
#     if img1.dim() == 4:
#         mse = torch.mean((img1 - img2) ** 2, (1, 2, 3), True)
#         mse[mse <= epsilon] = epsilon
#     else:
#         mse = ((img1 - img2) ** 2).mean()
#         mse = epsilon if mse <= epsilon else mse
#     PIXEL_MAX = 1
#     psnrs = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
#     return psnrs.mean()
# ################
#
# ###this is for l,ab,hintmask,g_ab
#
# def patch2img(patch_output, patch_size=16, out_shape=(224,224)):
#     """
#     Convert patch outputs [B, N_patches, C*patch_size*patch_size] to full image [B, C, H, W]
#     For ab channels, C=2, patch_output is [B, N, patch_size*patch_size*2]
#     """
#     B, N, vec = patch_output.shape
#     C = vec // (patch_size * patch_size)
#     H, W = out_shape
#     out = patch_output.reshape(B, N, C, patch_size, patch_size)
#     n_h = H // patch_size
#     n_w = W // patch_size
#     out = out.reshape(B, n_h, n_w, C, patch_size, patch_size)
#     out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
#     return out
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# ########################this is modified by me by yupp on 27th june


# File: utils.py

import io
import os
import random
from collections import defaultdict, deque
import datetime
import numpy as np
import torch
import torch.distributed as dist
from skimage.color import lab2rgb as sk_lab2rgb, rgb2lab as sk_rgb2lab
from timm.utils import get_state_dict
import torch.nn as nn
import time


# --------------------------------------------------------
# Funtions for Image-Color Space Conversion
# --------------------------------------------------------

def rgb2lab(rgb_norm, l_norm=100, ab_norm=110):
    # This function is a wrapper for skimage.color.rgb2lab
    # It handles the conversion from a normalized torch tensor [0, 1] to the LAB color space
    rgb_norm = rgb_norm.permute(0, 2, 3, 1)  # (B, H, W, C)
    img_lab = sk_rgb2lab(rgb_norm.detach().cpu().numpy())  # (B, H, W, 3)

    # Normalize L channel to [0, 1] and ab channels to [-1, 1]
    l_channel = img_lab[:, :, :, :1] / l_norm
    ab_channels = np.clip(img_lab[:, :, :, 1:] / ab_norm, -1, 1)

    img_lab_norm = np.concatenate((l_channel, ab_channels), axis=3)
    img_lab_norm = torch.from_numpy(img_lab_norm).float().permute(0, 3, 1, 2)
    return img_lab_norm.to(rgb_norm.device)


def lab2rgb(lab_norm, l_norm=100, ab_norm=110):
    # This function is a wrapper for skimage.color.lab2rgb
    # It converts a normalized LAB tensor back to a normalized RGB tensor [0, 1]
    lab_norm = lab_norm.permute(0, 2, 3, 1)  # (B, H, W, C)
    lab_unnorm = lab_norm.clone().detach().cpu().numpy()
    lab_unnorm[:, :, :, 0] *= l_norm
    lab_unnorm[:, :, :, 1:] *= ab_norm

    rgb_unnorm = sk_lab2rgb(lab_unnorm)  # (B, H, W, 3)
    rgb_norm = torch.from_numpy(rgb_unnorm).float().permute(0, 3, 1, 2)
    return rgb_norm.to(lab_norm.device)


def psnr(img1, img2):
    # Calculates the Peak Signal-to-Noise Ratio between two images
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# --------------------------------------------------------
# Funtions for Distributed Training
# --------------------------------------------------------

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# --------------------------------------------------------
# Functions for Logging and Metrics
# --------------------------------------------------------

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if not self.deque: return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.deque: return 0.0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0: return 0.0
        return self.total / self.count

    @property
    def max(self):
        if not self.deque: return 0.0
        return max(self.deque)

    @property
    def value(self):
        if not self.deque: return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("Please install tensorboard to use TensorboardLogger.")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step)

    def flush(self):
        self.writer.flush()


# --------------------------------------------------------
# Functions for Model Saving/Loading and LR Scheduling
# --------------------------------------------------------

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = os.path.join(args.output_dir, f'checkpoint-{epoch}.pth')
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'args': args,
    }
    torch.save(to_save, output_dir)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

        print(f"Resume checkpoint {args.resume}")
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not args.resume_weights_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print(f"With optim & sched!")


# def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0,
#                      warmup_steps=-1):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_steps > 0:
#         warmup_iters = warmup_steps
#
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
#
#     iters = np.arange(epochs * niter_per_ep - warmup_iters)
#     schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
#
#     schedule = np.concatenate((warmup_schedule, schedule))
#     assert len(schedule) == epochs * niter_per_ep
#     return schedule

##this is modify by me "14th ock 2025 for thesis"
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    # print is fine but optional:
    # print(f"Set warmup steps = {warmup_iters}")

    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters, dtype=np.float64)

    total = int(epochs) * int(niter_per_ep)
    remain = max(total - warmup_iters, 0)
    if remain == 0:
        schedule = warmup_schedule
    else:
        iters = np.arange(remain, dtype=np.float64)
        # cosine from base_value -> final_value
        schedule_core = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(remain - 1, 1)))
        schedule = np.concatenate((warmup_schedule, schedule_core), axis=0)

    # Lenient guard: clamp to exact length
    if len(schedule) > total:
        schedule = schedule[:total]
    elif len(schedule) < total:
        last = float(schedule[-1]) if len(schedule) else float(final_value)
        schedule = np.concatenate([schedule, np.full(total - len(schedule), last, dtype=np.float64)], axis=0)

    return schedule



def get_parameter_groups(model, weight_decay=1e-5):
    # This function is not used in the final version but kept for reference
    # It separates parameters into those with and without weight decay
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]


def seed_worker(worker_id):
    # Set seed for dataloader workers
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm
