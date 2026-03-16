# # # # # # --------------------------------------------------------
# # # # # # Based on BEiT, timm, DINO and DeiT code bases
# # # # # # https://github.com/microsoft/unilm/tree/master/beit
# # # # # # https://github.com/rwightman/pytorch-image-models/tree/master/timm
# # # # # # https://github.com/facebookresearch/deit
# # # # # # https://github.com/facebookresearch/dino
# # # # # # --------------------------------------------------------'
# # # # # import math
# # # # # import sys
# # # # # from typing import Iterable
# # # # #
# # # # # import torch
# # # # # from einops import rearrange
# # # # # from tqdm import tqdm
# # # # #
# # # # # import utils
# # # # # from losses import HuberLoss
# # # # # from utils import lab2rgb, psnr, rgb2lab
# # # # #
# # # # #
# # # # # def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
# # # # #                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
# # # # #                     log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
# # # # #                     wd_schedule_values=None, exp_name=None):
# # # # #     model.train()
# # # # #     metric_logger = utils.MetricLogger(delimiter="  ")
# # # # #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # # # #     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # # # #     header = 'Epoch: [{}]'.format(epoch)
# # # # #     print_freq = 10
# # # # #
# # # # #     # loss_func = nn.MSELoss()
# # # # #     loss_func = HuberLoss()
# # # # #
# # # # #     for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
# # # # #         if step % 100 == 0:
# # # # #             print(exp_name)
# # # # #         # assign learning rate & weight decay for each step
# # # # #         it = start_steps + step  # global training iteration
# # # # #         if lr_schedule_values is not None or wd_schedule_values is not None:
# # # # #             for i, param_group in enumerate(optimizer.param_groups):
# # # # #                 if lr_schedule_values is not None:
# # # # #                     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
# # # # #                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
# # # # #                     param_group["weight_decay"] = wd_schedule_values[it]
# # # # #
# # # # #         images, bool_hinted_pos = batch
# # # # #
# # # # #         images = images.to(device, non_blocking=True)
# # # # #         # bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
# # # # #         bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)
# # # # #
# # # # #         # Lab conversion and normalizatoin
# # # # #         images = rgb2lab(images, 50, 100, 110)  # l_cent, l_norm, ab_norm
# # # # #         B, C, H, W = images.shape
# # # # #         h, w = H // patch_size, W // patch_size
# # # # #
# # # # #         # import pdb; pdb.set_trace()
# # # # #         with torch.no_grad():
# # # # #             # calculate the predict label
# # # # #             images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# # # # #             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # # #
# # # # #         with torch.cuda.amp.autocast():
# # # # #             outputs = model(images, bool_hinted_pos)  # ! images has been changed (in-place ops)
# # # # #             outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # # #
# # # # #             # Loss is calculated only with the ab channels
# # # # #             loss = loss_func(input=outputs, target=labels[:, :, :, 1:])
# # # # #
# # # # #         loss_value = loss.item()
# # # # #
# # # # #         if not math.isfinite(loss_value):
# # # # #             print("Loss is {}, stopping training".format(loss_value))
# # # # #             sys.exit(1)
# # # # #
# # # # #         optimizer.zero_grad()
# # # # #         # this attribute is added by timm on one optimizer (adahessian)
# # # # #         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
# # # # #         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
# # # # #                                 parameters=model.parameters(), create_graph=is_second_order)
# # # # #         loss_scale_value = loss_scaler.state_dict()["scale"]
# # # # #
# # # # #         torch.cuda.synchronize()
# # # # #
# # # # #         metric_logger.update(loss=loss_value)
# # # # #         metric_logger.update(loss_scale=loss_scale_value)
# # # # #         min_lr = 10.
# # # # #         max_lr = 0.
# # # # #         for group in optimizer.param_groups:
# # # # #             min_lr = min(min_lr, group["lr"])
# # # # #             max_lr = max(max_lr, group["lr"])
# # # # #
# # # # #         metric_logger.update(lr=max_lr)
# # # # #         metric_logger.update(min_lr=min_lr)
# # # # #         weight_decay_value = None
# # # # #         for group in optimizer.param_groups:
# # # # #             if group["weight_decay"] > 0:
# # # # #                 weight_decay_value = group["weight_decay"]
# # # # #         metric_logger.update(weight_decay=weight_decay_value)
# # # # #         metric_logger.update(grad_norm=grad_norm)
# # # # #
# # # # #         if log_writer is not None:
# # # # #             log_writer.update(loss=loss_value, head="loss")
# # # # #             log_writer.update(loss_scale=loss_scale_value, head="opt")
# # # # #             log_writer.update(lr=max_lr, head="opt")
# # # # #             log_writer.update(min_lr=min_lr, head="opt")
# # # # #             log_writer.update(weight_decay=weight_decay_value, head="opt")
# # # # #             log_writer.update(grad_norm=grad_norm, head="opt")
# # # # #
# # # # #             log_writer.set_step()
# # # # #
# # # # #         if lr_scheduler is not None:
# # # # #             lr_scheduler.step_update(start_steps + step)
# # # # #     # gather the stats from all processes
# # # # #     metric_logger.synchronize_between_processes()
# # # # #     print("Averaged stats:", metric_logger)
# # # # #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# # # # #
# # # # #
# # # # # def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
# # # # #              patch_size: int = 16, log_writer=None, val_hint_list=[10]):
# # # # #     model.eval()
# # # # #     header = 'Validation'
# # # # #
# # # # #     psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
# # # # #     num_validated = 0
# # # # #     with torch.no_grad():
# # # # #         for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
# # # # #             # assign learning rate & weight decay for each step
# # # # #             images, bool_hints = batch
# # # # #             B, _, H, W = images.shape
# # # # #             h, w = H // patch_size, W // patch_size
# # # # #
# # # # #             images = images.to(device, non_blocking=True)
# # # # #             # Lab conversion and normalizatoin
# # # # #             images_lab = rgb2lab(images)
# # # # #             # calculate the predict label
# # # # #             images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# # # # #             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # # #
# # # # #             for idx, count in enumerate(val_hint_list):
# # # # #                 bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
# # # # #                 # bool_hint = bool_hints.to(device, non_blocking=True).to(torch.bool)
# # # # #
# # # # #                 with torch.cuda.amp.autocast():
# # # # #                     outputs = model(images_lab.clone(), bool_hint.clone())
# # # # #                     outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # # #
# # # # #                 pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
# # # # #                 pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
# # # # #                                           h=h, w=w, p1=patch_size, p2=patch_size)
# # # # #                 pred_imgs = lab2rgb(pred_imgs_lab)
# # # # #
# # # # #                 _psnr = psnr(images, pred_imgs) * B
# # # # #                 psnr_sum[count] += _psnr.item()
# # # # #             num_validated += B
# # # # #
# # # # #         psnr_avg = dict()
# # # # #         for count in val_hint_list:
# # # # #             psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated
# # # # #
# # # # #         torch.cuda.synchronize()
# # # # #
# # # # #         if log_writer is not None:
# # # # #             log_writer.update(head="psnr", **psnr_avg)
# # # # #     return psnr_avg
# # # # ############################################
# # # # ################################
# # # # #########################
# # # # import math
# # # # import sys
# # # # from typing import Iterable
# # # # import torch
# # # # from einops import rearrange
# # # # from tqdm import tqdm
# # # # import utils
# # # # from losses import HuberLoss
# # # # from utils import lab2rgb, psnr, rgb2lab
# # # #
# # # #
# # # # def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler):
# # # #     model.train()
# # # #
# # # #     loss_func = HuberLoss()
# # # #
# # # #     for step, batch in enumerate(data_loader):
# # # #         # Handle unexpected unpacking errors
# # # #         if len(batch) == 2:
# # # #             images, bool_hinted_pos = batch
# # # #         else:
# # # #             images, bool_hinted_pos, *_ = batch
# # # #
# # # #         images = images.to(device)
# # # #         bool_hinted_pos = bool_hinted_pos.to(device).to(torch.bool)
# # # #
# # # #         images = rgb2lab(images)
# # # #         B, C, H, W = images.shape
# # # #
# # # #         with torch.cuda.amp.autocast():
# # # #             outputs = model(images, bool_hinted_pos)
# # # #             loss = loss_func(outputs, images[:, :, :, 1:])
# # # #
# # # #         optimizer.zero_grad()
# # # #         loss_scaler(loss, optimizer)
# # # #
# # # #         print(f"Epoch [{epoch}] Step [{step}] Loss: {loss.item()}")
# # # #
# # # #
# # # # def validate(model, data_loader, device):
# # # #     model.eval()
# # # #
# # # #     psnr_sum = 0
# # # #     count = 0
# # # #
# # # #     with torch.no_grad():
# # # #         for batch in tqdm(data_loader, desc="Validation"):
# # # #             if len(batch) == 2:
# # # #                 images, bool_hints = batch
# # # #             else:
# # # #                 images, bool_hints, *_ = batch
# # # #
# # # #             images = images.to(device)
# # # #             outputs = model(images, bool_hints)
# # # #
# # # #             psnr_sum += psnr(images, outputs)
# # # #             count += 1
# # # #
# # # #     print(f"Validation PSNR: {psnr_sum / count}")
# # # # ##############################################################################
# # # # ##############################################################################
# # # # --------------------------------------------------------
# # # # Based on BEiT, timm, DINO and DeiT code bases
# # # # https://github.com/microsoft/unilm/tree/master/beit
# # # # https://github.com/rwightman/pytorch-image-models/tree/master/timm
# # # # https://github.com/facebookresearch/deit
# # # # https://github.com/facebookresearch/dino
# # # # --------------------------------------------------------'
# # # # # #from here this is previous code without any optimization loss
# # # # #############################################################################
# # # # ##############################################################################
# # # # import math
# # # # import sys
# # # # from typing import Iterable
# # # #
# # # # import torch
# # # # from einops import rearrange
# # # # from tqdm import tqdm
# # # #
# # # # import utils
# # # # from losses import HuberLoss
# # # # from utils import lab2rgb, psnr, rgb2lab
# # # #
# # # #
# # # # def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
# # # #                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
# # # #                     log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
# # # #                     wd_schedule_values=None, exp_name=None):
# # # #     model.train()
# # # #     metric_logger = utils.MetricLogger(delimiter="  ")
# # # #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # # #     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # # #     header = 'Epoch: [{}]'.format(epoch)
# # # #     print_freq = 10
# # # #
# # # #     # loss_func = nn.MSELoss()
# # # #     loss_func = HuberLoss()
# # # #
# # # #     for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
# # # #         if step % 100 == 0:
# # # #             print(exp_name)
# # # #         # assign learning rate & weight decay for each step
# # # #         it = start_steps + step  # global training iteration
# # # #         if lr_schedule_values is not None or wd_schedule_values is not None:
# # # #             for i, param_group in enumerate(optimizer.param_groups):
# # # #                 if lr_schedule_values is not None:
# # # #                     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
# # # #                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
# # # #                     param_group["weight_decay"] = wd_schedule_values[it]
# # # #
# # # #         images, bool_hinted_pos = batch
# # # #
# # # #         images = images.to(device, non_blocking=True)
# # # #         # bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
# # # #         bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)
# # # #
# # # #         # Lab conversion and normalizatoin
# # # #         images = rgb2lab(images, 50, 100, 110)  # l_cent, l_norm, ab_norm
# # # #         B, C, H, W = images.shape
# # # #         h, w = H // patch_size, W // patch_size
# # # #
# # # #         # import pdb; pdb.set_trace()
# # # #         with torch.no_grad():
# # # #             # calculate the predict label
# # # #             images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# # # #             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # #
# # # #         with torch.cuda.amp.autocast():
# # # #             outputs = model(images, bool_hinted_pos)  # ! images has been changed (in-place ops)
# # # #             outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # #
# # # #             # Loss is calculated only with the ab channels
# # # #             loss = loss_func(input=outputs, target=labels[:, :, :, 1:])
# # # #
# # # #         loss_value = loss.item()
# # # #
# # # #         if not math.isfinite(loss_value):
# # # #             print("Loss is {}, stopping training".format(loss_value))
# # # #             sys.exit(1)
# # # #
# # # #         optimizer.zero_grad()
# # # #         # this attribute is added by timm on one optimizer (adahessian)
# # # #         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
# # # #         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
# # # #                                 parameters=model.parameters(), create_graph=is_second_order)
# # # #         loss_scale_value = loss_scaler.state_dict()["scale"]
# # # #
# # # #         torch.cuda.synchronize()
# # # #
# # # #         metric_logger.update(loss=loss_value)
# # # #         metric_logger.update(loss_scale=loss_scale_value)
# # # #         min_lr = 10.
# # # #         max_lr = 0.
# # # #         for group in optimizer.param_groups:
# # # #             min_lr = min(min_lr, group["lr"])
# # # #             max_lr = max(max_lr, group["lr"])
# # # #
# # # #         metric_logger.update(lr=max_lr)
# # # #         metric_logger.update(min_lr=min_lr)
# # # #         weight_decay_value = None
# # # #         for group in optimizer.param_groups:
# # # #             if group["weight_decay"] > 0:
# # # #                 weight_decay_value = group["weight_decay"]
# # # #         metric_logger.update(weight_decay=weight_decay_value)
# # # #         metric_logger.update(grad_norm=grad_norm)
# # # #
# # # #         if log_writer is not None:
# # # #             log_writer.update(loss=loss_value, head="loss")
# # # #             log_writer.update(loss_scale=loss_scale_value, head="opt")
# # # #             log_writer.update(lr=max_lr, head="opt")
# # # #             log_writer.update(min_lr=min_lr, head="opt")
# # # #             log_writer.update(weight_decay=weight_decay_value, head="opt")
# # # #             log_writer.update(grad_norm=grad_norm, head="opt")
# # # #
# # # #             log_writer.set_step()
# # # #
# # # #         if lr_scheduler is not None:
# # # #             lr_scheduler.step_update(start_steps + step)
# # # #     # gather the stats from all processes
# # # #     metric_logger.synchronize_between_processes()
# # # #     print("Averaged stats:", metric_logger)
# # # #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# # # #
# # # #
# # # # def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
# # # #              patch_size: int = 16, log_writer=None, val_hint_list=[10]):
# # # #     model.eval()
# # # #     header = 'Validation'
# # # #
# # # #     psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
# # # #     num_validated = 0
# # # #     with torch.no_grad():
# # # #         for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
# # # #             # assign learning rate & weight decay for each step
# # # #             images, bool_hints = batch
# # # #             B, _, H, W = images.shape
# # # #             h, w = H // patch_size, W // patch_size
# # # #
# # # #             images = images.to(device, non_blocking=True)
# # # #             # Lab conversion and normalizatoin
# # # #             images_lab = rgb2lab(images)
# # # #             # calculate the predict label
# # # #             images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# # # #             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # #
# # # #             for idx, count in enumerate(val_hint_list):
# # # #                 bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
# # # #                 # bool_hint = bool_hints.to(device, non_blocking=True).to(torch.bool)
# # # #
# # # #                 with torch.cuda.amp.autocast():
# # # #                     outputs = model(images_lab.clone(), bool_hint.clone())
# # # #                     outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # #
# # # #                 pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
# # # #                 pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
# # # #                                           h=h, w=w, p1=patch_size, p2=patch_size)
# # # #                 pred_imgs = lab2rgb(pred_imgs_lab)
# # # #
# # # #                 _psnr = psnr(images, pred_imgs) * B
# # # #                 psnr_sum[count] += _psnr.item()
# # # #             num_validated += B
# # # #
# # # #         psnr_avg = dict()
# # # #         for count in val_hint_list:
# # # #             psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated
# # # #
# # # #         torch.cuda.synchronize()
# # # #
# # # #         if log_writer is not None:
# # # #             log_writer.update(head="psnr", **psnr_avg)
# # # #     return psnr_avg
# # #
# # # # ################################
# # # #This is for optimization of loss lambda =3 and so on if required
# # # import math
# # # import sys
# # # from typing import Iterable
# # #
# # # import torch
# # # from einops import rearrange
# # # from tqdm import tqdm
# # #
# # # import utils
# # # from utils import lab2rgb, psnr, rgb2lab
# # #
# # #
# # # def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
# # #                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
# # #                     log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
# # #                     wd_schedule_values=None, exp_name=None):
# # #     model.train()
# # #     metric_logger = utils.MetricLogger(delimiter="  ")
# # #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # #     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # #     header = 'Epoch: [{}]'.format(epoch)
# # #     print_freq = 10
# # #
# # #     loss_func = torch.nn.MSELoss(reduction='none')  # We'll weight it manually
# # #     lambda_tpr = 3.0  # Higher value prioritizes foreground (TPR)
# # #     #lambda_tpr = 4.0  # Higher value prioritizes foreground (TPR)
# # #     #lambda_tpr = 2.5  # Higher value prioritizes foreground (TPR)
# # #     for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
# # #         if step % 100 == 0:
# # #             print(exp_name)
# # #         it = start_steps + step
# # #
# # #         if lr_schedule_values is not None or wd_schedule_values is not None:
# # #             for i, param_group in enumerate(optimizer.param_groups):
# # #                 if lr_schedule_values is not None:
# # #                     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
# # #                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
# # #                     param_group["weight_decay"] = wd_schedule_values[it]
# # #
# # #         images, bool_hinted_pos = batch
# # #         images = images.to(device, non_blocking=True)
# # #         bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)
# # #
# # #         images = rgb2lab(images, 50, 100, 110)
# # #         B, C, H, W = images.shape
# # #         h, w = H // patch_size, W // patch_size
# # #
# # #         with torch.no_grad():
# # #             images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# # #             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # #
# # #         with torch.cuda.amp.autocast():
# # #             outputs = model(images, bool_hinted_pos)
# # #             outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # #
# # #             gt_ab = labels[:, :, :, 1:]
# # #             print("gt_ab:", gt_ab)
# # #
# # #             mse_pixelwise = loss_func(outputs, gt_ab)
# # #
# # #             # gt_mask = (gt_ab.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
# # #
# # #             # weight = gt_mask * lambda_tpr + (1.0 - gt_mask)
# # #
# # #             gt_mask = (gt_ab.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
# # #             print("gt_mask:", gt_mask)
# # #             ####please check these logic whether it~s oimpact on loss or tpr thinhgs or not
# # #             weight = gt_mask * lambda_tpr + (1.0 - gt_mask)
# # #             # Set the threshold to a large value to print the full tensor
# # #             torch.set_printoptions(threshold=10000)  # or higher if needed[1][7]
# # #
# # #             print("weight:", weight)
# # #
# # #             #print("weight:", weight)
# # #
# # #             weighted_loss = mse_pixelwise * weight
# # #             loss = weighted_loss.mean()
# # #
# # #         loss_value = loss.item()
# # #
# # #         if not math.isfinite(loss_value):
# # #             print("Loss is {}, stopping training".format(loss_value))
# # #             sys.exit(1)
# # #
# # #         optimizer.zero_grad()
# # #         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
# # #         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
# # #                                 parameters=model.parameters(), create_graph=is_second_order)
# # #         loss_scale_value = loss_scaler.state_dict()["scale"]
# # #
# # #         torch.cuda.synchronize()
# # #
# # #         metric_logger.update(loss=loss_value)
# # #         metric_logger.update(loss_scale=loss_scale_value)
# # #         min_lr = 10.
# # #         max_lr = 0.
# # #         for group in optimizer.param_groups:
# # #             min_lr = min(min_lr, group["lr"])
# # #             max_lr = max(max_lr, group["lr"])
# # #
# # #         metric_logger.update(lr=max_lr)
# # #         metric_logger.update(min_lr=min_lr)
# # #         weight_decay_value = None
# # #         for group in optimizer.param_groups:
# # #             if group["weight_decay"] > 0:
# # #                 weight_decay_value = group["weight_decay"]
# # #         metric_logger.update(weight_decay=weight_decay_value)
# # #         metric_logger.update(grad_norm=grad_norm)
# # #
# # #         if log_writer is not None:
# # #             log_writer.update(loss=loss_value, head="loss")
# # #             log_writer.update(loss_scale=loss_scale_value, head="opt")
# # #             log_writer.update(lr=max_lr, head="opt")
# # #             log_writer.update(min_lr=min_lr, head="opt")
# # #             log_writer.update(weight_decay=weight_decay_value, head="opt")
# # #             log_writer.update(grad_norm=grad_norm, head="opt")
# # #             log_writer.set_step()
# # #
# # #         if lr_scheduler is not None:
# # #             lr_scheduler.step_update(start_steps + step)
# # #
# # #     metric_logger.synchronize_between_processes()
# # #     print("Averaged stats:", metric_logger)
# # #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# # #
# # #
# # # @torch.no_grad()
# # # def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
# # #              patch_size: int = 16, log_writer=None, val_hint_list=[10]):
# # #     model.eval()
# # #     header = 'Validation'
# # #
# # #     from einops import rearrange
# # #     from utils import lab2rgb, psnr, rgb2lab
# # #
# # #     psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
# # #     num_validated = 0
# # #     for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
# # #         images, bool_hints = batch
# # #         B, _, H, W = images.shape
# # #         h, w = H // patch_size, W // patch_size
# # #
# # #         images = images.to(device, non_blocking=True)
# # #         images_lab = rgb2lab(images)
# # #         images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# # #         labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # #
# # #         for idx, count in enumerate(val_hint_list):
# # #             bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
# # #
# # #             with torch.cuda.amp.autocast():
# # #                 outputs = model(images_lab.clone(), bool_hint.clone())
# # #                 outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # #
# # #             pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
# # #             pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
# # #                                       h=h, w=w, p1=patch_size, p2=patch_size)
# # #             pred_imgs = lab2rgb(pred_imgs_lab)
# # #
# # #             _psnr = psnr(images, pred_imgs) * B
# # #             psnr_sum[count] += _psnr.item()
# # #         num_validated += B
# # #
# # #     psnr_avg = dict()
# # #     for count in val_hint_list:
# # #         psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated
# # #
# # #     torch.cuda.synchronize()
# # #
# # #     if log_writer is not None:
# # #         log_writer.update(head="psnr", **psnr_avg)
# # #
# # #     return psnr_avg
# # #
# # #
# # # ''''
# # # ###############################################################################################################
# # # ###############################################################################################################
# # # ###############################################################################################################
# # # # Patch by yuvi
# # # ###############################################################################################################
# # # import math
# # # import sys
# # # import torch
# # # import torch.nn.functional as F
# # # from tqdm import tqdm
# # # from einops import rearrange
# # #
# # # def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
# # #                     max_norm=0, log_writer=None, start_steps=0,
# # #                     lr_schedule_values=None, wd_schedule_values=None,
# # #                     patch_size=16, exp_name=""):
# # #     model.train()
# # #     metric_logger = utils.MetricLogger(delimiter="  ")
# # #     header = f'Train Epoch: [{epoch}]'
# # #
# # #     for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
# # #         samples = batch['L'].to(device, non_blocking=True)
# # #         targets = batch['ab'].to(device, non_blocking=True)
# # #         hint = batch['hint_ab'].to(device, non_blocking=True)
# # #         mask = batch['hint_mask'].to(device, non_blocking=True)
# # #
# # #         step = data_iter_step + start_steps
# # #         if lr_schedule_values is not None or wd_schedule_values is not None:
# # #             for i, param_group in enumerate(optimizer.param_groups):
# # #                 if lr_schedule_values is not None:
# # #                     param_group['lr'] = lr_schedule_values[step] * param_group.get("lr_scale", 1.0)
# # #                 if wd_schedule_values is not None and param_group['weight_decay'] > 0:
# # #                     param_group['weight_decay'] = wd_schedule_values[step]
# # #
# # #         with torch.amp.autocast(device_type='cuda'):
# # #             output = model(samples, hint, mask)
# # #
# # #             # [B, 196, 512] → [B, 14, 14, 512] → [B, 512, 14, 14]
# # #             output = output.permute(0, 2, 1).reshape(-1, 512, 14, 14)
# # #
# # #             # Now reshape to match GT: [B, 2, 224, 224]
# # #             output = rearrange(output, 'b (c p1 p2) h w -> b c (h p1) (w p2)', c=2, p1=16, p2=16)
# # #
# # #             loss = F.smooth_l1_loss(output, targets)
# # #
# # #         loss_value = loss.item()
# # #         optimizer.zero_grad()
# # #         loss.backward()
# # #         if max_norm is not None:
# # #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
# # #         optimizer.step()
# # #
# # #         torch.cuda.synchronize()
# # #
# # #         metric_logger.update(loss=loss_value)
# # #         if log_writer is not None:
# # #             log_writer.update(head='train', loss=loss_value, step=step)
# # #
# # #     metric_logger.synchronize_between_processes()
# # #     print("Averaged stats:", metric_logger)
# # #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# # #
# # # def validate(model, data_loader_val, device, patch_size, log_writer=None, step=None):
# # #     model.eval()
# # #     metric_logger = utils.MetricLogger(delimiter="  ")
# # #     header = 'Validation:'
# # #
# # #     with torch.amp.autocast(device_type='cuda'):
# # #         for batch in metric_logger.log_every(data_loader_val, 10, header):
# # #             samples = batch['L'].to(device, non_blocking=True)
# # #             targets = batch['ab'].to(device, non_blocking=True)
# # #             hint = batch['hint_ab'].to(device, non_blocking=True)
# # #             mask = batch['hint_mask'].to(device, non_blocking=True)
# # #
# # #             with torch.cuda.amp.autocast():
# # #                 output = model(samples, hint, mask)
# # #
# # #                 output = output.permute(0, 2, 1).reshape(-1, 512, 14, 14)  # Match train
# # #                 output = rearrange(output, 'b (c p1 p2) h w -> b c (h p1) (w p2)', c=2, p1=16, p2=16)
# # #
# # #                 loss = F.smooth_l1_loss(output, targets)
# # #
# # #             loss_value = loss.item()
# # #             metric_logger.update(loss=loss_value)
# # #             if log_writer is not None:
# # #                 # log_writer.add_scalar('val/loss', loss_value, global_step=None)
# # #                 log_writer.update(head='val', loss=loss_value, step=step)
# # #
# # #     metric_logger.synchronize_between_processes()
# # #     print("Validation stats:", metric_logger)
# # #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# # # ###############################################################################################################
# # # ###############################################################################################################
# # # ###############################################################################################################
# # # '''
# # # ###############################################################################################################
# # # ###############################################################################################################
# # # ###########################################################################################################
# # #
# # # # ###############################################
# # # # # Adaptive Lambda Weighting for Training Script
# # # # ###############################################
# # # # import math
# # # # import sys
# # # # from typing import Iterable
# # # #
# # # # import torch
# # # # from einops import rearrange
# # # # from tqdm import tqdm
# # # #
# # # # import utils
# # # # from utils import lab2rgb, psnr, rgb2lab
# # # #
# # # #
# # # # def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
# # # #                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
# # # #                     log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
# # # #                     wd_schedule_values=None, exp_name=None):
# # # #     model.train()
# # # #     metric_logger = utils.MetricLogger(delimiter="  ")
# # # #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # # #     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # # #     header = 'Epoch: [{}]'.format(epoch)
# # # #     print_freq = 10
# # # #
# # # #     loss_func = torch.nn.MSELoss(reduction='none')
# # # #     lambda_tpr = 2.5
# # # #
# # # #     for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
# # # #         if step % 100 == 0:
# # # #             print(exp_name)
# # # #         it = start_steps + step
# # # #
# # # #         if lr_schedule_values is not None or wd_schedule_values is not None:
# # # #             for i, param_group in enumerate(optimizer.param_groups):
# # # #                 if lr_schedule_values is not None:
# # # #                     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
# # # #                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
# # # #                     param_group["weight_decay"] = wd_schedule_values[it]
# # # #
# # # #         images, bool_hinted_pos = batch
# # # #         images = images.to(device, non_blocking=True)
# # # #         bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)
# # # #
# # # #         images = rgb2lab(images, 50, 100, 110)
# # # #         B, C, H, W = images.shape
# # # #         h, w = H // patch_size, W // patch_size
# # # #
# # # #         with torch.no_grad():
# # # #             images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# # # #             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # #
# # # #         with torch.cuda.amp.autocast():
# # # #             outputs = model(images, bool_hinted_pos)
# # # #             outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # #
# # # #             gt_ab = labels[:, :, :, 1:]
# # # #             mse_pixelwise = loss_func(outputs, gt_ab)
# # # #
# # # #             gt_mask = (gt_ab.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
# # # #             weight = gt_mask * lambda_tpr + (1.0 - gt_mask)
# # # #             weighted_loss = mse_pixelwise * weight
# # # #             loss = weighted_loss.mean()
# # # #
# # # #         loss_value = loss.item()
# # # #
# # # #         if not math.isfinite(loss_value):
# # # #             print("Loss is {}, stopping training".format(loss_value))
# # # #             sys.exit(1)
# # # #
# # # #         optimizer.zero_grad()
# # # #         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
# # # #         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
# # # #                                 parameters=model.parameters(), create_graph=is_second_order)
# # # #         loss_scale_value = loss_scaler.state_dict()["scale"]
# # # #
# # # #         torch.cuda.synchronize()
# # # #
# # # #         metric_logger.update(loss=loss_value)
# # # #         metric_logger.update(loss_scale=loss_scale_value)
# # # #         min_lr = 10.
# # # #         max_lr = 0.
# # # #         for group in optimizer.param_groups:
# # # #             min_lr = min(min_lr, group["lr"])
# # # #             max_lr = max(max_lr, group["lr"])
# # # #
# # # #         metric_logger.update(lr=max_lr)
# # # #         metric_logger.update(min_lr=min_lr)
# # # #         weight_decay_value = None
# # # #         for group in optimizer.param_groups:
# # # #             if group["weight_decay"] > 0:
# # # #                 weight_decay_value = group["weight_decay"]
# # # #         metric_logger.update(weight_decay=weight_decay_value)
# # # #         metric_logger.update(grad_norm=grad_norm)
# # # #
# # # #         if log_writer is not None:
# # # #             log_writer.update(loss=loss_value, head="loss")
# # # #             log_writer.update(loss_scale=loss_scale_value, head="opt")
# # # #             log_writer.update(lr=max_lr, head="opt")
# # # #             log_writer.update(min_lr=min_lr, head="opt")
# # # #             log_writer.update(weight_decay=weight_decay_value, head="opt")
# # # #             log_writer.update(grad_norm=grad_norm, head="opt")
# # # #             log_writer.set_step()
# # # #
# # # #         if lr_scheduler is not None:
# # # #             lr_scheduler.step_update(start_steps + step)
# # # #
# # # #     metric_logger.synchronize_between_processes()
# # # #     print("Averaged stats:", metric_logger)
# # # #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# # # #
# # # #
# # # # @torch.no_grad()
# # # # def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
# # # #              patch_size: int = 16, log_writer=None, val_hint_list=[10]):
# # # #     model.eval()
# # # #     header = 'Validation'
# # # #
# # # #     psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
# # # #     num_validated = 0
# # # #     for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
# # # #         images, bool_hints = batch
# # # #         B, _, H, W = images.shape
# # # #         h, w = H // patch_size, W // patch_size
# # # #
# # # #         images = images.to(device, non_blocking=True)
# # # #         images_lab = rgb2lab(images)
# # # #         images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# # # #         labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # #
# # # #         for idx, count in enumerate(val_hint_list):
# # # #             bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
# # # #
# # # #             with torch.cuda.amp.autocast():
# # # #                 outputs = model(images_lab.clone(), bool_hint.clone())
# # # #                 outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# # # #
# # # #             pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
# # # #             pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
# # # #                                       h=h, w=w, p1=patch_size, p2=patch_size)
# # # #             pred_imgs = lab2rgb(pred_imgs_lab)
# # # #
# # # #             _psnr = psnr(images, pred_imgs) * B
# # # #             psnr_sum[count] += _psnr.item()
# # # #         num_validated += B
# # # #
# # # #     psnr_avg = dict()
# # # #     for count in val_hint_list:
# # # #         psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated
# # # #
# # # #     torch.cuda.synchronize()
# # # #
# # # #     if log_writer is not None:
# # # #         log_writer.update(head="psnr", **psnr_avg)
# # # #
# # # #     return psnr_avg
# # # #
# # # #
# #
# #
# # ###################this is done by me for patch wise square ,horizontal selection
# # # import math
# # # import sys
# # # from typing import Iterable
# # #
# # # import torch
# # # from einops import rearrange
# # # from tqdm import tqdm
# # #
# # # import utils
# # # from utils import lab2rgb, psnr, rgb2lab
# # #
# # # def train_one_epoch(
# # #     model: torch.nn.Module,
# # #     data_loader: Iterable,
# # #     optimizer: torch.optim.Optimizer,
# # #     device: torch.device,
# # #     epoch: int,
# # #     loss_scaler,
# # #     max_norm: float = 0,
# # #     patch_size: int = 16,
# # #     log_writer=None,
# # #     lr_scheduler=None,
# # #     start_steps=None,
# # #     lr_schedule_values=None,
# # #     wd_schedule_values=None,
# # #     exp_name=None,
# # # ):
# # #     model.train()
# # #     metric_logger = utils.MetricLogger(delimiter="  ")
# # #     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
# # #     metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
# # #     header = f"Epoch: [{epoch}]"
# # #     print_freq = 10
# # #
# # #     loss_func = torch.nn.MSELoss(reduction="none")  # We'll weight it manually
# # #
# # #     # Adaptive lambda_tpr parameters
# # #     lambda_tpr = 3.0
# # #     target_tpr = 0.92  # set as you wish
# # #     tpr_momentum = 0.96
# # #
# # #     running_tpr = target_tpr
# # #
# # #     for step, (imgs, hint_masks, ab_hints, img_names) in enumerate(
# # #         metric_logger.log_every(data_loader, print_freq, header)
# # #     ):
# # #         it = start_steps + step if start_steps is not None else step
# # #
# # #         imgs = imgs.to(device, non_blocking=True)  # [B, 3, H, W] or [B, 1, H, W] if grayscale
# # #         hint_masks = hint_masks.to(device, non_blocking=True)  # [B, n_patches, patch_area]
# # #         ab_hints = ab_hints.to(device, non_blocking=True)      # [B, n_patches, patch_area, 2]
# # #
# # #         # 1. Get [B, n_patches] patch-wise mask for model
# # #         patch_mask_for_model = (hint_masks.sum(dim=-1) > 0).float()  # [B, n_patches]
# # #
# # #         # 2. Get [B, n_patches, patch_area, 1] for pixel-wise foreground/background weighting
# # #         mask_for_weight = hint_masks.unsqueeze(-1)  # [B, n_patches, patch_area, 1]
# # #
# # #         # 3. Optionally, compute running TPR for adaptive weighting (e.g. hint coverage per batch)
# # #         batch_tpr = mask_for_weight.float().mean().item()
# # #         running_tpr = tpr_momentum * running_tpr + (1 - tpr_momentum) * batch_tpr
# # #
# # #         # 4. Adaptively adjust lambda_tpr if running_tpr falls below target
# # #         if running_tpr < target_tpr:
# # #             lambda_tpr = min(lambda_tpr + 0.1, 5.0)  # increase to prioritize fg
# # #         else:
# # #             lambda_tpr = max(lambda_tpr - 0.05, 2.0) # decay slowly
# # #
# # #         with torch.cuda.amp.autocast():
# # #             # Model expects: imgs, patch_mask_for_model
# # #             outputs = model(imgs, patch_mask_for_model)  # [B, n_patches, patch_area, 2] (pred ab)
# # #             mse_pixelwise = loss_func(outputs, ab_hints)  # [B, n_patches, patch_area, 2]
# # #             # Weight: fg gets lambda_tpr, bg gets 1
# # #             weight = mask_for_weight * lambda_tpr + (1.0 - mask_for_weight)
# # #             weighted_loss = mse_pixelwise * weight
# # #             loss = weighted_loss.mean()
# # #
# # #         loss_value = loss.item()
# # #         if not math.isfinite(loss_value):
# # #             print(f"Loss is {loss_value}, stopping training")
# # #             sys.exit(1)
# # #
# # #         optimizer.zero_grad()
# # #         is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
# # #         grad_norm = loss_scaler(
# # #             loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order
# # #         )
# # #         loss_scale_value = loss_scaler.state_dict()["scale"]
# # #         torch.cuda.synchronize()
# # #
# # #         metric_logger.update(loss=loss_value)
# # #         metric_logger.update(loss_scale=loss_scale_value)
# # #         metric_logger.update(fg_weight=lambda_tpr)
# # #         metric_logger.update(batch_tpr=batch_tpr)
# # #         metric_logger.update(running_tpr=running_tpr)
# # #
# # #         min_lr, max_lr = 10.0, 0.0
# # #         for group in optimizer.param_groups:
# # #             min_lr = min(min_lr, group["lr"])
# # #             max_lr = max(max_lr, group["lr"])
# # #         metric_logger.update(lr=max_lr)
# # #         metric_logger.update(min_lr=min_lr)
# # #         weight_decay_value = None
# # #         for group in optimizer.param_groups:
# # #             if group["weight_decay"] > 0:
# # #                 weight_decay_value = group["weight_decay"]
# # #         metric_logger.update(weight_decay=weight_decay_value)
# # #         metric_logger.update(grad_norm=grad_norm)
# # #
# # #         if log_writer is not None:
# # #             log_writer.update(loss=loss_value, head="loss")
# # #             log_writer.update(loss_scale=loss_scale_value, head="opt")
# # #             log_writer.update(lr=max_lr, head="opt")
# # #             log_writer.update(min_lr=min_lr, head="opt")
# # #             log_writer.update(weight_decay=weight_decay_value, head="opt")
# # #             log_writer.update(grad_norm=grad_norm, head="opt")
# # #             log_writer.update(fg_weight=lambda_tpr, head="opt")
# # #             log_writer.update(batch_tpr=batch_tpr, head="opt")
# # #             log_writer.update(running_tpr=running_tpr, head="opt")
# # #             log_writer.set_step()
# # #
# # #         if lr_scheduler is not None:
# # #             lr_scheduler.step_update(it)
# # #
# # #     metric_logger.synchronize_between_processes()
# # #     print("Averaged stats:", metric_logger)
# # #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# # #
# # # @torch.no_grad()
# # # def validate(
# # #     model: torch.nn.Module,
# # #     data_loader: Iterable,
# # #     device: torch.device,
# # #     patch_size: int = 16,
# # #     log_writer=None,
# # #     val_hint_list=[10],
# # # ):
# # #     model.eval()
# # #     header = "Validation"
# # #
# # #     psnr_sum = dict(zip(val_hint_list, [0.0] * len(val_hint_list)))
# # #     num_validated = 0
# # #     for step, (imgs, hint_masks, ab_hints, img_names) in tqdm(
# # #         enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)
# # #     ):
# # #         imgs = imgs.to(device, non_blocking=True)
# # #         hint_masks = hint_masks.to(device, non_blocking=True)
# # #
# # #         patch_mask_for_model = (hint_masks.sum(dim=-1) > 0).float()
# # #
# # #         # Lab conversion, if needed, can go here (optional)
# # #         with torch.cuda.amp.autocast():
# # #             outputs = model(imgs, patch_mask_for_model)
# # #         # Assume outputs are ab channels
# # #         # Reconstruct images and compute PSNR
# # #         # TODO: Adapt for your format: outputs = [B, n_patches, patch_area, 2]
# # #         # You'll need to "unpatchify" if you want full images for PSNR
# # #         # For now, this just counts batch size for avg
# # #         psnr_sum[val_hint_list[0]] += 0.0  # Dummy, replace with your PSNR calculation
# # #         num_validated += imgs.shape[0]
# # #
# # #     psnr_avg = {}
# # #     for count in val_hint_list:
# # #         psnr_avg[f"psnr@{count}"] = psnr_sum[count] / max(num_validated, 1)
# # #
# # #     torch.cuda.synchronize()
# # #     if log_writer is not None:
# # #         log_writer.update(head="psnr", **psnr_avg)
# # #     return psnr_avg
# # ##############
# #
# # ######this is for only patches in .npz as hints selected area as 1 or else 0
# #
# # # --------------------------------------------------------
# # # Based on BEiT, timm, DINO and DeiT code bases
# # # https://github.com/microsoft/unilm/tree/master/beit
# # # https://github.com/rwightman/pytorch-image-models/tree/master/timm
# # # https://github.com/facebookresearch/deit
# # # https://github.com/facebookresearch/dino
# # # --------------------------------------------------------
# # # import math
# # # import sys
# # # from typing import Iterable
# # #
# # # import torch
# # # from einops import rearrange
# # # from tqdm import tqdm
# # #
# # # import utils
# # # from losses import HuberLoss
# # # from utils import lab2rgb, psnr, rgb2lab
# # #
# # #
# # # def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
# # #                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
# # #                     log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
# # #                     wd_schedule_values=None, exp_name=None):
# # #     model.train()
# # #     metric_logger = utils.MetricLogger(delimiter="  ")
# # #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # #     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# # #     header = 'Epoch: [{}]'.format(epoch)
# # #     print_freq = 10
# # #
# # #     loss_func = HuberLoss()
# # #
# # #     for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
# # #         if step % 100 == 0:
# # #             print(exp_name)
# # #
# # #         it = start_steps + step if start_steps is not None else step
# # #         if lr_schedule_values is not None or wd_schedule_values is not None:
# # #             for i, param_group in enumerate(optimizer.param_groups):
# # #                 if lr_schedule_values is not None:
# # #                     param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
# # #                 if wd_schedule_values is not None and param_group.get("weight_decay", 0) > 0:
# # #                     param_group["weight_decay"] = wd_schedule_values[it]
# # #
# # #         samples = samples.to(device, non_blocking=True)
# # #         targets = targets.to(device, non_blocking=True)
# # #
# # #         with torch.cuda.amp.autocast():
# # #             img_l, hint_mask = samples  # assuming your Dataset returns a tuple
# # #             outputs = model(img_l, hint_mask)
# # #
# # #             #outputs = model(samples)
# # #             loss = loss_func(outputs, targets)
# # #
# # #         loss_value = loss.item()
# # #
# # #         if not math.isfinite(loss_value):
# # #             print("Loss is {}, stopping training".format(loss_value))
# # #             sys.exit(1)
# # #
# # #         optimizer.zero_grad()
# # #         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
# # #         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
# # #                                 parameters=model.parameters(), create_graph=is_second_order)
# # #         loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)
# # #
# # #         torch.cuda.synchronize()
# # #
# # #         metric_logger.update(loss=loss_value)
# # #         metric_logger.update(loss_scale=loss_scale_value)
# # #         min_lr, max_lr = float('inf'), 0.
# # #         for group in optimizer.param_groups:
# # #             min_lr = min(min_lr, group.get("lr", 0.))
# # #             max_lr = max(max_lr, group.get("lr", 0.))
# # #
# # #         metric_logger.update(lr=max_lr)
# # #         metric_logger.update(min_lr=min_lr)
# # #         weight_decay_value = next((group["weight_decay"] for group in optimizer.param_groups if group.get("weight_decay", 0) > 0), 0.)
# # #         metric_logger.update(weight_decay=weight_decay_value)
# # #         metric_logger.update(grad_norm=grad_norm)
# # #
# # #         if log_writer is not None:
# # #             log_writer.update(loss=loss_value, head="loss")
# # #             log_writer.update(loss_scale=loss_scale_value, head="opt")
# # #             log_writer.update(lr=max_lr, head="opt")
# # #             log_writer.update(min_lr=min_lr, head="opt")
# # #             log_writer.update(weight_decay=weight_decay_value, head="opt")
# # #             log_writer.update(grad_norm=grad_norm, head="opt")
# # #             log_writer.set_step()
# # #
# # #         if lr_scheduler is not None:
# # #             lr_scheduler.step_update(it)
# # #
# # #     metric_logger.synchronize_between_processes()
# # #     print("Averaged stats:", metric_logger)
# # #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# # #
# # #
# # # def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
# # #              patch_size: int = 16, log_writer=None, val_hint_list=[10]):
# # #     model.eval()
# # #     header = 'Validation'
# # #
# # #     psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
# # #     num_validated = 0
# # #
# # #     with torch.no_grad():
# # #         for step, (samples, targets) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
# # #             samples = samples.to(device, non_blocking=True)
# # #             targets = targets.to(device, non_blocking=True)
# # #
# # #             with torch.cuda.amp.autocast():
# # #                 outputs = model(samples)
# # #
# # #             pred_imgs_lab = torch.cat((samples[:, :1, :, :], outputs), dim=1)
# # #             pred_imgs = lab2rgb(pred_imgs_lab)
# # #             targets_rgb = lab2rgb(torch.cat((samples[:, :1, :, :], targets), dim=1))
# # #             _psnr = psnr(targets_rgb, pred_imgs) * samples.size(0)
# # #
# # #             psnr_sum[val_hint_list[0]] += _psnr.item()
# # #             num_validated += samples.size(0)
# # #
# # #     psnr_avg = {f'psnr@{val_hint_list[0]}': psnr_sum[val_hint_list[0]] / num_validated}
# # #
# # #     torch.cuda.synchronize()
# # #     if log_writer is not None:
# # #         log_writer.update(head="psnr", **psnr_avg)
# # #     return psnr_avg
# # #
# # ####################################################this is done by me for hints selected area as 1
# #
# #
# # # engine.py (fully updated for patchwise training)
# #
# # import math
# # import sys
# # import torch
# # from typing import Iterable
# # from tqdm import tqdm
# #
# # import torch.nn.functional as F
# #
# # def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
# #                     device: torch.device, epoch: int, loss_scaler, patch_size: int = 224,
# #                     print_freq: int = 10, exp_name: str = "", log_writer=None, criterion=None):
# #     model.train()
# #     metric_logger = tqdm(data_loader, desc=f"[TRAIN] Epoch {epoch}")
# #     total_loss = 0.0
# #
# #     for step, (img_l, hint_mask, hint_ab, target_ab) in enumerate(metric_logger):
# #         img_l = img_l.to(device, non_blocking=True)
# #         hint_mask = hint_mask.to(device, non_blocking=True)
# #         hint_ab = hint_ab.to(device, non_blocking=True)
# #         target_ab = target_ab.to(device, non_blocking=True)
# #
# #         input_tensor = torch.cat([img_l, hint_mask, hint_ab], dim=1)  # [B, 4, 224, 224]
# #
# #         with torch.cuda.amp.autocast():
# #             B = input_tensor.shape[0]
# #             mask = torch.ones(B, 196).to(input_tensor.device)  # 14x14=196 patches for 224x224 inputs
# #             output_ab = model(input_tensor, mask=mask)
# #             loss = criterion(output_ab, target_ab)
# #
# #         optimizer.zero_grad()
# #         loss_scaler(loss, optimizer, parameters=model.parameters())
# #         total_loss += loss.item()
# #
# #         if log_writer is not None:
# #             global_step = epoch * len(data_loader) + step
# #             log_writer.add_scalar("train/loss", loss.item(), global_step)
# #
# #         if step % print_freq == 0:
# #             metric_logger.set_postfix(loss=loss.item())
# #
# #     avg_loss = total_loss / len(data_loader)
# #     print(f"[TRAIN] Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
# #     return {"loss": avg_loss}
# #
# #
# # def validate(model, criterion, data_loader, device, args=None, log_writer=None):
# #     model.eval()
# #     total_loss = 0.0
# #     metric_logger = tqdm(data_loader, desc="[VAL]")
# #     with torch.no_grad():
# #         for step, (img_l, hint_mask, hint_ab, target_ab) in enumerate(metric_logger):
# #             img_l = img_l.to(device, non_blocking=True)
# #             hint_mask = hint_mask.to(device, non_blocking=True)
# #             hint_ab = hint_ab.to(device, non_blocking=True)
# #             target_ab = target_ab.to(device, non_blocking=True)
# #
# #             input_tensor = torch.cat([img_l, hint_mask, hint_ab], dim=1)
# #             B = input_tensor.shape[0]
# #             mask = torch.ones(B, 196).to(input_tensor.device)  # 14x14=196 patches for 224x224 inputs
# #             output_ab = model(input_tensor, mask=mask)
# #             #output_ab = model(input_tensor)
# #             loss = criterion(output_ab, target_ab)
# #
# #             total_loss += loss.item()
# #             if step % 10 == 0:
# #                 metric_logger.set_postfix(val_loss=loss.item())
# #
# #     avg_loss = total_loss / len(data_loader)
# #     print(f"[VAL] Avg Loss = {avg_loss:.4f}")
# #
# #     if log_writer is not None:
# #         log_writer.add_scalar("val/loss", avg_loss)
# #
# #     return {"loss": avg_loss}
# ##this is modified by me for L,ab,hint_mask, G_ab
#
#
# #######################
# ###here i have just did comment for checking other , this is done by me 24thjune below
# ##################################################################################################
# # ##################################################################################################
# #
# # ###############################################################################
# # ###############################################################################
# # # --------------------------------------------------------
# # # Based on BEiT, timm, DINO and DeiT code bases
# # # https://github.com/microsoft/unilm/tree/master/beit
# # # https://github.com/rwightman/pytorch-image-models/tree/master/timm
# # # https://github.com/facebookresearch/deit
# # # https://github.com/facebookresearch/dino
# # # --------------------------------------------------------'
# # import math
# # import sys
# # from typing import Iterable
# #
# # import torch
# # from einops import rearrange
# # from tqdm import tqdm
# #
# # import utils
# # from losses import HuberLoss
# # from utils import lab2rgb, psnr, rgb2lab
# #
# #
# # def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
# #                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
# #                     log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
# #                     wd_schedule_values=None, exp_name=None):
# #     model.train()
# #     metric_logger = utils.MetricLogger(delimiter="  ")
# #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# #     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# #     header = 'Epoch: [{}]'.format(epoch)
# #     print_freq = 10
# #
# #     # loss_func = nn.MSELoss()
# #     loss_func = HuberLoss()
# #
# #     for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
# #         if step % 100 == 0:
# #             print(exp_name)
# #         # assign learning rate & weight decay for each step
# #         it = start_steps + step  # global training iteration
# #         if lr_schedule_values is not None or wd_schedule_values is not None:
# #             for i, param_group in enumerate(optimizer.param_groups):
# #                 if lr_schedule_values is not None:
# #                     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
# #                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
# #                     param_group["weight_decay"] = wd_schedule_values[it]
# #
# #         images, bool_hinted_pos = batch
# #
# #         images = images.to(device, non_blocking=True)
# #         # bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
# #         bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)
# #
# #         # Lab conversion and normalizatoin
# #         images = rgb2lab(images, 50, 100, 110)  # l_cent, l_norm, ab_norm
# #         B, C, H, W = images.shape
# #         h, w = H // patch_size, W // patch_size
# #
# #         # import pdb; pdb.set_trace()
# #         with torch.no_grad():
# #             # calculate the predict label
# #             images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# #             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# #
# #         with torch.cuda.amp.autocast():
# #             outputs = model(images, bool_hinted_pos)  # ! images has been changed (in-place ops)
# #             outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# #
# #             # Loss is calculated only with the ab channels
# #             loss = loss_func(input=outputs, target=labels[:, :, :, 1:])
# #
# #         loss_value = loss.item()
# #
# #         if not math.isfinite(loss_value):
# #             print("Loss is {}, stopping training".format(loss_value))
# #             sys.exit(1)
# #
# #         optimizer.zero_grad()
# #         # this attribute is added by timm on one optimizer (adahessian)
# #         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
# #         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
# #                                 parameters=model.parameters(), create_graph=is_second_order)
# #         loss_scale_value = loss_scaler.state_dict()["scale"]
# #
# #         torch.cuda.synchronize()
# #
# #         metric_logger.update(loss=loss_value)
# #         metric_logger.update(loss_scale=loss_scale_value)
# #         min_lr = 10.
# #         max_lr = 0.
# #         for group in optimizer.param_groups:
# #             min_lr = min(min_lr, group["lr"])
# #             max_lr = max(max_lr, group["lr"])
# #
# #         metric_logger.update(lr=max_lr)
# #         metric_logger.update(min_lr=min_lr)
# #         weight_decay_value = None
# #         for group in optimizer.param_groups:
# #             if group["weight_decay"] > 0:
# #                 weight_decay_value = group["weight_decay"]
# #         metric_logger.update(weight_decay=weight_decay_value)
# #         metric_logger.update(grad_norm=grad_norm)
# #
# #         if log_writer is not None:
# #             log_writer.update(loss=loss_value, head="loss")
# #             log_writer.update(loss_scale=loss_scale_value, head="opt")
# #             log_writer.update(lr=max_lr, head="opt")
# #             log_writer.update(min_lr=min_lr, head="opt")
# #             log_writer.update(weight_decay=weight_decay_value, head="opt")
# #             log_writer.update(grad_norm=grad_norm, head="opt")
# #
# #             log_writer.set_step()
# #
# #         if lr_scheduler is not None:
# #             lr_scheduler.step_update(start_steps + step)
# #     # gather the stats from all processes
# #     metric_logger.synchronize_between_processes()
# #     print("Averaged stats:", metric_logger)
# #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# #
# #
# # def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
# #              patch_size: int = 16, log_writer=None, val_hint_list=[10]):
# #     model.eval()
# #     header = 'Validation'
# #
# #     psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
# #     num_validated = 0
# #     with torch.no_grad():
# #         for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
# #             # assign learning rate & weight decay for each step
# #             images, bool_hints = batch
# #             B, _, H, W = images.shape
# #             h, w = H // patch_size, W // patch_size
# #
# #             images = images.to(device, non_blocking=True)
# #             # Lab conversion and normalizatoin
# #             images_lab = rgb2lab(images)
# #             # calculate the predict label
# #             images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# #             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# #
# #             for idx, count in enumerate(val_hint_list):
# #                 bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)
# #                 # bool_hint = bool_hints.to(device, non_blocking=True).to(torch.bool)
# #
# #                 with torch.cuda.amp.autocast():
# #                     outputs = model(images_lab.clone(), bool_hint.clone())
# #                     outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
# #
# #                 pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
# #                 pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
# #                                           h=h, w=w, p1=patch_size, p2=patch_size)
# #                 pred_imgs = lab2rgb(pred_imgs_lab)
# #
# #                 _psnr = psnr(images, pred_imgs) * B
# #                 psnr_sum[count] += _psnr.item()
# #             num_validated += B
# #
# #         psnr_avg = dict()
# #         for count in val_hint_list:
# #             psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated
# #
# #         torch.cuda.synchronize()
# #
# #         if log_writer is not None:
# #             log_writer.update(head="psnr", **psnr_avg)
# #     return psnr_avg
#
#
#
# # ########this is below for l,ab,hint_mask,G_ab
# #
# # import math
# # import sys
# # from typing import Iterable
# #
# # import torch
# # from einops import rearrange
# # from tqdm import tqdm
# #
# # import utils
# # from losses import HuberLoss
# # from utils import lab2rgb, psnr, rgb2lab
# #
# #
# # def train_one_epoch(
# #     model: torch.nn.Module,
# #     data_loader: Iterable,
# #     optimizer: torch.optim.Optimizer,
# #     device: torch.device,
# #     epoch: int,
# #     loss_scaler,
# #     max_norm: float = 0,
# #     patch_size: int = 16,
# #     log_writer=None,
# #     lr_scheduler=None,
# #     start_steps=None,
# #     lr_schedule_values=None,
# #     wd_schedule_values=None,
# #     exp_name=None
# # ):
# #     model.train()
# #     metric_logger = utils.MetricLogger(delimiter="  ")
# #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# #     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# #     header = 'Epoch: [{}]'.format(epoch)
# #     print_freq = 10
# #
# #     loss_func = HuberLoss()
# #
# #     for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
# #         if step % 100 == 0:
# #             print(exp_name)
# #         # assign learning rate & weight decay for each step
# #         it = start_steps + step if start_steps is not None else step  # global training iteration
# #         if lr_schedule_values is not None or wd_schedule_values is not None:
# #             for i, param_group in enumerate(optimizer.param_groups):
# #                 if lr_schedule_values is not None:
# #                     param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
# #                 if wd_schedule_values is not None and param_group.get("weight_decay", 0) > 0:
# #                     param_group["weight_decay"] = wd_schedule_values[it]
# #
# #         # --- PATCHWISE NPZ SUPPORT ---
# #         if isinstance(batch, (list, tuple)) and len(batch) == 2:
# #             inputs, targets = batch
# #             # If extra tuple, (inputs, targets, ..), just take first two
# #             if isinstance(inputs, (list, tuple)) and hasattr(inputs[0], "shape"):
# #                 # Could be dataset returns ((inputs, ...), targets)
# #                 inputs = inputs[0]
# #         else:
# #             # fallback for legacy dataloader (unlikely needed)
# #             raise ValueError("Unexpected batch structure: {}".format(type(batch)))
# #
# #         # If you ever use datasets returning (_, _, name), handle it here!
# #         # E.g., ((inputs, targets), name)
# #
# #         # Move to device
# #         inputs = inputs.to(device, non_blocking=True)
# #         targets = targets.to(device, non_blocking=True)
# #
# #         # No need for lab conversion: we assume inputs are already [4,224,224] (L, ab_hint, hint_mask)
# #         # targets are [2,224,224] (GT ab)
# #         with torch.cuda.amp.autocast():
# #             outputs = model(inputs)
# #             # outputs: [B, 2, 224, 224] or similar
# #
# #             # Huber loss on ab channels
# #             loss = loss_func(outputs, targets)
# #
# #         loss_value = loss.item()
# #         if not math.isfinite(loss_value):
# #             print("Loss is {}, stopping training".format(loss_value))
# #             sys.exit(1)
# #
# #         optimizer.zero_grad()
# #         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
# #         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
# #                                 parameters=model.parameters(), create_graph=is_second_order)
# #         loss_scale_value = loss_scaler.state_dict()["scale"]
# #
# #         torch.cuda.synchronize()
# #
# #         metric_logger.update(loss=loss_value)
# #         metric_logger.update(loss_scale=loss_scale_value)
# #         min_lr = 10.
# #         max_lr = 0.
# #         for group in optimizer.param_groups:
# #             min_lr = min(min_lr, group["lr"])
# #             max_lr = max(max_lr, group["lr"])
# #
# #         metric_logger.update(lr=max_lr)
# #         metric_logger.update(min_lr=min_lr)
# #         weight_decay_value = None
# #         for group in optimizer.param_groups:
# #             if group.get("weight_decay", 0) > 0:
# #                 weight_decay_value = group["weight_decay"]
# #         metric_logger.update(weight_decay=weight_decay_value)
# #         metric_logger.update(grad_norm=grad_norm)
# #
# #         if log_writer is not None:
# #             log_writer.update(loss=loss_value, head="loss")
# #             log_writer.update(loss_scale=loss_scale_value, head="opt")
# #             log_writer.update(lr=max_lr, head="opt")
# #             log_writer.update(min_lr=min_lr, head="opt")
# #             log_writer.update(weight_decay=weight_decay_value, head="opt")
# #             log_writer.update(grad_norm=grad_norm, head="opt")
# #             log_writer.set_step()
# #
# #         if lr_scheduler is not None:
# #             lr_scheduler.step_update(it)
# #     # gather the stats from all processes
# #     metric_logger.synchronize_between_processes()
# #     print("Averaged stats:", metric_logger)
# #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# #
# #
# # def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
# #              patch_size: int = 16, log_writer=None, val_hint_list=[10]):
# #     model.eval()
# #     header = 'Validation'
# #
# #     psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
# #     num_validated = 0
# #     with torch.no_grad():
# #         for step, batch in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
# #             # PATCHWISE NPZ VALIDATION SUPPORT (assume .npz loader gives (inputs, targets))
# #             if isinstance(batch, (list, tuple)) and len(batch) == 2:
# #                 inputs, targets = batch
# #                 inputs = inputs.to(device, non_blocking=True)
# #                 targets = targets.to(device, non_blocking=True)
# #                 # Model outputs ab channels only. If you want to compute PSNR, convert back to RGB!
# #                 outputs = model(inputs)
# #                 # You may need to add back the L channel to compute Lab->RGB if required.
# #                 # For now, just compute MSE/PSNR on ab channels as in training.
# #                 _psnr = psnr(targets, outputs) * targets.size(0)
# #                 psnr_sum[val_hint_list[0]] += _psnr.item()
# #                 num_validated += targets.size(0)
# #             else:
# #                 # fallback for legacy dataloader (not needed for .npz mode)
# #                 raise ValueError("Unexpected batch structure in validation.")
# #
# #         psnr_avg = dict()
# #         for count in val_hint_list:
# #             psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated if num_validated > 0 else 0.
# #     return psnr_avg
#
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import time
# from collections import defaultdict
#
# def mask_to_patch_mask(mask, patch_size=16):
#     # mask: [B, 1, 224, 224], float {0,1}
#     # Output: [B, n_patches] (flattened)
#     B, _, H, W = mask.shape
#     assert H % patch_size == 0 and W % patch_size == 0, f'Input {H},{W} not divisible by {patch_size}'
#     nH, nW = H // patch_size, W // patch_size
#     mask_unfold = F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size)
#     patch_mask = (mask_unfold > 0.5).float().view(B, -1)  # [B, n_patches]
#     return patch_mask
#
# def train_one_epoch(
#     model, data_loader, optimizer, device, epoch, loss_scaler,
#     clip_grad=None, log_writer=None,
#     start_steps=0, lr_schedule_values=None, wd_schedule_values=None,
#     patch_size=16, exp_name=None
# ):
#     model.train()
#     metric_logger = defaultdict(float)
#     n = 0
#     for step, (inputs, targets) in enumerate(data_loader):
#         # -- Patch-wise dense region hint setup --
#         inputs = inputs.to(device, non_blocking=True)    # [B, 4, 224, 224]
#         targets = targets.to(device, non_blocking=True)  # [B, 2, 224, 224]
#         mask = inputs[:, 3:4, :, :]                     # [B, 1, 224, 224]
#         patch_mask = mask_to_patch_mask(mask, patch_size=patch_size)
#         model_inputs = inputs                           # [B, 4, 224, 224] -- DO NOT SLICE
#
#         with torch.cuda.amp.autocast():
#             outputs = model(model_inputs, patch_mask)   # [B, n_patches, patch_vec]
#             out_pred = patch2img(outputs, patch_size=patch_size, out_shape=targets.shape[-2:])
#             # [B, 2, H, W] (matches target)
#             print("out_pred.shape:", out_pred.shape)
#             print("targets.shape:", targets.shape)
#
#             loss = F.huber_loss(out_pred, targets)
#
#         loss_value = loss.item()
#         n += 1
#         metric_logger['loss'] += loss_value
#
#         optimizer.zero_grad()
#         loss_scaler.scale(loss).backward()
#         if clip_grad is not None:
#             loss_scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
#         loss_scaler.step(optimizer)
#         loss_scaler.update()
#
#         if log_writer is not None and (step % 10 == 0):
#             log_writer.add_scalar('train/loss', loss_value, start_steps + step)
#
#     # Average
#     metric_logger = {k: v / n for k, v in metric_logger.items()}
#     return metric_logger
#
# @torch.no_grad()
# def validate(model, data_loader, device, patch_size=16, log_writer=None, val_hint_list=[0]):
#     model.eval()
#     metric_logger = defaultdict(float)
#     n = 0
#     for step, (inputs, targets) in enumerate(data_loader):
#         inputs = inputs.to(device, non_blocking=True)    # [B, 4, 224, 224]
#         targets = targets.to(device, non_blocking=True)  # [B, 2, 224, 224]
#         mask = inputs[:, 3:4, :, :]                     # [B, 1, 224, 224]
#         patch_mask = mask_to_patch_mask(mask, patch_size=patch_size)
#         model_inputs = inputs                           # [B, 4, 224, 224] -- DO NOT SLICE
#
#         with torch.cuda.amp.autocast():
#             outputs = model(model_inputs, patch_mask)
#             out_pred = patch2img(outputs, patch_size=patch_size, out_shape=targets.shape[-2:])
#             loss = F.huber_loss(out_pred, targets)
#
#         loss_value = loss.item()
#         n += 1
#         metric_logger['loss'] += loss_value
#
#         if log_writer is not None and (step % 10 == 0):
#             log_writer.add_scalar('val/loss', loss_value, step)
#
#     metric_logger = {k: v / n for k, v in metric_logger.items()}
#     return metric_logger
#
#
# import torch.nn.functional as F
#
# def patch2img(patches, patch_size=16, out_shape=(224, 224)):
#     """
#     Accepts patches of shape:
#         [B, n_patches, C]
#         [B, C, nH, nW]
#     Returns:
#         [B, C, H, W]
#     """
#     if patches.dim() == 3:
#         # [B, n_patches, C] -> [B, C, nH, nW]
#         B, n_patches, C = patches.shape
#         nH = out_shape[0] // patch_size
#         nW = out_shape[1] // patch_size
#         assert nH * nW == n_patches, f"Mismatch: n_patches={n_patches}, grid={nH}x{nW}"
#         patches = patches.permute(0, 2, 1).contiguous().view(B, C, nH, nW)
#     elif patches.dim() == 4:
#         # [B, C, nH, nW]
#         B, C, nH, nW = patches.shape
#     else:
#         raise ValueError(f"patches must be 3D or 4D, got {patches.shape}")
#     # Upsample to full resolution
#     patches = F.interpolate(patches, size=out_shape, mode='bilinear', align_corners=False)
#     return patches
##############################################################################
########the above is for l,ab,hintmask, g_ab

########################################################################################
#######################################
##########the below is for patch hints directory h20 and patches wise

######

##################################################################################################################
###############################################################################################################
###########################################################################################################
'''
###############################################
# Adaptive Lambda Weighting for Training Script
###############################################
import math
import sys
from typing import Iterable

import torch
from einops import rearrange
from tqdm import tqdm

import utils
from utils import lab2rgb, psnr, rgb2lab


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, exp_name=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = torch.nn.MSELoss(reduction='none')
    lambda_tpr = 2.5

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % 100 == 0:
            print(exp_name)
        it = start_steps + step

        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_hinted_pos = batch
        images = images.to(device, non_blocking=True)
        bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)

        images = rgb2lab(images, 50, 100, 110)
        B, C, H, W = images.shape
        h, w = H // patch_size, W // patch_size

        with torch.no_grad():
            images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

        with torch.cuda.amp.autocast():
            outputs = model(images, bool_hinted_pos)
            outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            gt_ab = labels[:, :, :, 1:]
            mse_pixelwise = loss_func(outputs, gt_ab)

            gt_mask = (gt_ab.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
            weight = gt_mask * lambda_tpr + (1.0 - gt_mask)
            weighted_loss = mse_pixelwise * weight
            loss = weighted_loss.mean()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
             patch_size: int = 16, log_writer=None, val_hint_list=[10]):
    model.eval()
    header = 'Validation'

    psnr_sum = dict(zip(val_hint_list, [0.] * len(val_hint_list)))
    num_validated = 0
    for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
        images, bool_hints = batch
        B, _, H, W = images.shape
        h, w = H // patch_size, W // patch_size

        images = images.to(device, non_blocking=True)
        images_lab = rgb2lab(images)
        images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

        for idx, count in enumerate(val_hint_list):
            bool_hint = bool_hints[:, idx].to(device, non_blocking=True).flatten(1).to(torch.bool)

            with torch.cuda.amp.autocast():
                outputs = model(images_lab.clone(), bool_hint.clone())
                outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)

            pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
            pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                      h=h, w=w, p1=patch_size, p2=patch_size)
            pred_imgs = lab2rgb(pred_imgs_lab)

            _psnr = psnr(images, pred_imgs) * B
            psnr_sum[count] += _psnr.item()
        num_validated += B

    psnr_avg = dict()
    for count in val_hint_list:
        psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated

    torch.cuda.synchronize()

    if log_writer is not None:
        log_writer.update(head="psnr", **psnr_avg)

    return psnr_avg

'''


##this below is mask from resnet50unet yuppfor patches
#
# # File: engine.py
# import math
# import sys
# from typing import Iterable
#
# import torch
# import torch.nn.functional as F  # --- Make sure F is imported
# from einops import rearrange
# from tqdm import tqdm
#
# import utils
# from utils import lab2rgb, psnr, rgb2lab
#
#
# def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
#                     log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
#                     wd_schedule_values=None, exp_name=None):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     #metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     #metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     # Change metric logger precision for LR:
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
#     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
#
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10
#
#     loss_func = torch.nn.MSELoss(reduction='none')
#     lambda_tpr = 2.5
#
#     for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         if step % 100 == 0:
#             if exp_name:
#                 print(exp_name)
#         it = start_steps + step
#
#         if lr_schedule_values is not None or wd_schedule_values is not None:
#             for i, param_group in enumerate(optimizer.param_groups):
#                 if lr_schedule_values is not None:
#                     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
#                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                     param_group["weight_decay"] = wd_schedule_values[it]
#
#         # --- MODIFICATION: Unpack image and patch-level mask ---
#         images, bool_hinted_pos = batch
#         images = images.to(device, non_blocking=True)
#         bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)
#
#         # Convert to LAB colorspace
#         #images_lab = rgb2lab(images, 50, 100, 110)
#         images_lab = rgb2lab(images)
#         B, C, H, W = images_lab.shape
#         h, w = H // patch_size, W // patch_size
#
#         # --- FIX #1: Prepare the 4-channel input for the model ---
#         # Create a full-resolution pixel mask from the patch-level boolean mask
#         side = int(math.sqrt(bool_hinted_pos.shape[1]))  # Should be 14 for 224/16
#         mask_2d = bool_hinted_pos.view(B, 1, side, side).float()
#         full_mask_for_input = F.interpolate(mask_2d, size=(H, W), mode='nearest')
#
#         # Mask the 'a' and 'b' channels of the LAB image where there are no hints
#         full_mask_for_ab_channels = (full_mask_for_input > 0.5)
#         images_lab_masked = images_lab.clone()  # Use a clone to preserve original for labels
#         images_lab_masked[:, 1, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
#         images_lab_masked[:, 2, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
#
#         # Concatenate the masked LAB image with the full mask to create a 4-channel tensor
#         # This is the input the `4ch` model expects
#         input_4ch_tensor = torch.cat((images_lab_masked, full_mask_for_input), dim=1)
#         # --- END FIX #1 ---
#
#         with torch.no_grad():
#             # Create labels from the original, unmasked LAB image
#             images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
#             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
#
#         with torch.cuda.amp.autocast():
#             # Pass the 4D input tensor and the original patch-level mask to the model
#             outputs = model(input_4ch_tensor, bool_hinted_pos)
#
#             # --- FIX #2: Correctly rearrange the model's 3D output ---
#             # Model output is [B, n, features], where features = patch_area * 2.
#             # We explicitly tell einops that 'c' (channels) is 2.
#             outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size, c=2)
#             # --- END FIX #2 ---
#
#             gt_ab = labels[:, :, :, 1:]
#             mse_pixelwise = loss_func(outputs, gt_ab)
#
#             gt_mask = (gt_ab.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
#             weight = gt_mask * lambda_tpr + (1.0 - gt_mask)
#             weighted_loss = mse_pixelwise * weight
#             loss = weighted_loss.mean()
#
#         loss_value = loss.item()
#
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)
#
#         optimizer.zero_grad()
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
#                                 parameters=model.parameters(), create_graph=is_second_order)
#         # ... (rest of the function is the same) ...
#
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#
#
# @torch.no_grad()
# def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
#              patch_size: int = 16, log_writer=None, val_hint_list=[10]):
#     model.eval()
#     header = 'Validation'
#
#     psnr_sum = {val_num: 0. for val_num in val_hint_list}
#     num_validated = 0
#     for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
#         images, bool_hints_batch = batch  # bool_hints_batch is [B, num_levels, n_patches]
#         B, _, H, W = images.shape
#         h, w = H // patch_size, W // patch_size
#
#         images = images.to(device, non_blocking=True)
#         images_lab = rgb2lab(images)
#         images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
#         labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
#
#         for idx, count in enumerate(val_hint_list):
#             bool_hint = bool_hints_batch[:, idx].to(device, non_blocking=True).to(torch.bool)
#
#             # --- FIX #3: Apply the same input logic for validation ---
#             side = int(math.sqrt(bool_hint.shape[1]))
#             mask_2d = bool_hint.view(B, 1, side, side).float()
#             full_mask_for_input = F.interpolate(mask_2d, size=(H, W), mode='nearest')
#
#             val_images_lab_masked = images_lab.clone()
#             full_mask_for_ab_channels = (full_mask_for_input > 0.5)
#             val_images_lab_masked[:, 1, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
#             val_images_lab_masked[:, 2, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
#
#             val_input_4ch_tensor = torch.cat((val_images_lab_masked, full_mask_for_input), dim=1)
#             # --- END FIX #3 ---
#
#             with torch.cuda.amp.autocast():
#                 outputs = model(val_input_4ch_tensor, bool_hint)
#                 outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size, c=2)
#
#             pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
#             pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
#                                       h=h, w=w, p1=patch_size, p2=patch_size)
#             pred_imgs = lab2rgb(pred_imgs_lab)
#
#             _psnr = psnr(images, pred_imgs) * B
#             psnr_sum[count] += _psnr.item()
#         num_validated += B
#
#     psnr_avg = dict()
#     for count in val_hint_list:
#         psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated
#
#     torch.cuda.synchronize()
#
#     if log_writer is not None:
#         log_writer.update(head="psnr", **psnr_avg)
#
#     return psnr_avg


# #############learning rate fix
# # File: engine.py
# import math
# import sys
# from typing import Iterable
#
# import torch
# import torch.nn.functional as F
# from einops import rearrange
# from tqdm import tqdm
#
# import utils
# from utils import lab2rgb, psnr, rgb2lab
#
# def train_one_epoch(
#     model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
#     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
#     log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
#     wd_schedule_values=None, exp_name=None
# ):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
#     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
#     metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))
#     metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10
#
#     loss_func = torch.nn.MSELoss(reduction='none')
#     lambda_tpr = 2.5
#
#     for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         if step % 100 == 0 and exp_name:
#             print(f"[DEBUG] {exp_name} | Epoch: {epoch}, Step: {step}")
#         it = start_steps + step if start_steps is not None else step
#
#         if lr_schedule_values is not None or wd_schedule_values is not None:
#             for i, param_group in enumerate(optimizer.param_groups):
#                 if lr_schedule_values is not None:
#                     param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
#                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                     param_group["weight_decay"] = wd_schedule_values[it]
#
#         images, bool_hinted_pos = batch
#         images = images.to(device, non_blocking=True)
#         bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)
#
#         # Convert to LAB color space
#         images_lab = rgb2lab(images)
#         B, C, H, W = images_lab.shape
#         h, w = H // patch_size, W // patch_size
#
#         # Prepare hint mask upsampled to image size
#         side = int(math.sqrt(bool_hinted_pos.shape[1]))
#         mask_2d = bool_hinted_pos.view(B, 1, side, side).float()
#         full_mask_for_input = F.interpolate(mask_2d, size=(H, W), mode='nearest')
#         full_mask_for_ab_channels = (full_mask_for_input > 0.5)
#
#         images_lab_masked = images_lab.clone()
#         images_lab_masked[:, 1, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
#         images_lab_masked[:, 2, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
#         input_4ch_tensor = torch.cat((images_lab_masked, full_mask_for_input), dim=1)
#
#         with torch.no_grad():
#             images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
#             labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
#
#         with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
#             outputs = model(input_4ch_tensor, bool_hinted_pos)
#             outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size, c=2)
#
#             gt_ab = labels[:, :, :, 1:]
#             mse_pixelwise = loss_func(outputs, gt_ab)
#             gt_mask = (gt_ab.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
#             weight = gt_mask * lambda_tpr + (1.0 - gt_mask)
#             weighted_loss = mse_pixelwise * weight
#             loss = weighted_loss.mean()
#
#         loss_value = loss.item()
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)
#
#         # ---- DEBUGGING: Update logger with current lr, loss, grad_norm
#         current_lr = optimizer.param_groups[0]["lr"]
#         current_min_lr = min([pg["lr"] for pg in optimizer.param_groups])
#         metric_logger.update(lr=current_lr, min_lr=current_min_lr, loss=loss_value)
#
#         optimizer.zero_grad()
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
#                                 parameters=model.parameters(), create_graph=is_second_order)
#         if grad_norm is not None:
#             metric_logger.update(grad_norm=grad_norm if isinstance(grad_norm, float) else grad_norm.item())
#
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#
# @torch.no_grad()
# def validate(
#     model: torch.nn.Module, data_loader: Iterable, device: torch.device,
#     patch_size: int = 16, log_writer=None, val_hint_list=[10]
# ):
#     model.eval()
#     header = 'Validation'
#     psnr_sum = {val_num: 0. for val_num in val_hint_list}
#     num_validated = 0
#
#     for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
#         images, bool_hints_batch = batch  # bool_hints_batch is [B, num_levels, n_patches]
#         B, _, H, W = images.shape
#         h, w = H // patch_size, W // patch_size
#
#         images = images.to(device, non_blocking=True)
#         images_lab = rgb2lab(images)
#         images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
#         labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size)
#
#         for idx, count in enumerate(val_hint_list):
#             bool_hint = bool_hints_batch[:, idx].to(device, non_blocking=True).to(torch.bool)
#             side = int(math.sqrt(bool_hint.shape[1]))
#             mask_2d = bool_hint.view(B, 1, side, side).float()
#             full_mask_for_input = F.interpolate(mask_2d, size=(H, W), mode='nearest')
#
#             val_images_lab_masked = images_lab.clone()
#             full_mask_for_ab_channels = (full_mask_for_input > 0.5)
#             val_images_lab_masked[:, 1, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
#             val_images_lab_masked[:, 2, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
#
#             val_input_4ch_tensor = torch.cat((val_images_lab_masked, full_mask_for_input), dim=1)
#             with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
#                 outputs = model(val_input_4ch_tensor, bool_hint)
#                 outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size, c=2)
#
#             pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
#             pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
#                                       h=h, w=w, p1=patch_size, p2=patch_size)
#             pred_imgs = lab2rgb(pred_imgs_lab)
#
#             _psnr = psnr(images, pred_imgs) * B
#             psnr_sum[count] += _psnr.item()
#         num_validated += B
#
#     psnr_avg = dict()
#     for count in val_hint_list:
#         psnr_avg[f'psnr@{count}'] = psnr_sum[count] / num_validated
#
#     torch.cuda.synchronize()
#
#     if log_writer is not None:
#         log_writer.update(head="psnr", **psnr_avg)
#
#     return psnr_avg


#####################################################################################################


###thesios sub

############# learning rate fix + robust batch handling
# File: engine.py
import math
import sys
from typing import Iterable, Tuple, Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

import utils
from utils import lab2rgb, psnr, rgb2lab


# -----------------------------
# Helpers (safe, revert-proof)
# -----------------------------
def _as_tensor(x) -> torch.Tensor:
    """
    Convert x to torch.Tensor if it isn't already.
    Keeps dtype/shape if it's a tensor; otherwise uses torch.as_tensor(x).
    """
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _stack_if_list(x) -> torch.Tensor:
    """
    If x is a list/tuple of *per-sample* items, stack along dim=0.
    If x is already a Tensor, return as-is.

    NOTE: do NOT pass already-batched PAIRS like [images_BCHW, hints_BN] to this.
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty list/tuple encountered in batch; cannot stack.")
        # If this looks like a nested batched pair (two batched tensors), caller must unpack first.
        if len(x) == 2 and all(isinstance(t, torch.Tensor) for t in x):
            raise RuntimeError(
                "Nested batched pair passed to _stack_if_list. Unpack first: got shapes "
                f"{tuple(x[0].shape)} and {tuple(x[1].shape)}"
            )
        return torch.stack([_as_tensor(y) for y in x], dim=0)
    return _as_tensor(x)


def _maybe_split_batched_pair(obj):
    """
    If obj is a (list/tuple) of length >= 2 containing two batched tensors
    with matching batch dimension, return (images, hints). Otherwise return None.
    """
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        a, b = obj[0], obj[1]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dim() >= 1 and b.dim() >= 1 and a.shape[0] == b.shape[0]:
                return a, b
    return None


def _unpack_batch_train(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expected train batch content:
      images: [B, 3, H, W]  (RGB)
      bool_hinted_pos: [B, Npatches] (bool/0-1), flattened per-patch hint grid

    Supports:
      - dict: keys 'image'/'images' and 'bool_hinted_pos'/'hints'/'hint_mask_flat'
      - tuple/list: (images, hints, *extras)
      - nested: ((images, hints), extra)   <-- common source of the previous crash
    """
    # dict form
    if isinstance(batch, dict):
        images = batch.get('image', batch.get('images', None))
        hinted = (batch.get('bool_hinted_pos', None) or
                  batch.get('hints', None) or
                  batch.get('hint_mask_flat', None))
        if images is None or hinted is None:
            raise ValueError(f"Dict batch missing required keys. Found keys: {list(batch.keys())}")
        return _as_tensor(images), _as_tensor(hinted)

    # tuple/list forms
    if not isinstance(batch, (list, tuple)) or len(batch) < 1:
        raise ValueError(f"Unexpected batch structure: {type(batch)}")

    first = batch[0]
    second = batch[1] if len(batch) > 1 else None

    # Case: nested ((images, hints), extra)
    pair = _maybe_split_batched_pair(first)
    if pair is not None:
        images, hinted = pair
        return _as_tensor(images), _as_tensor(hinted)

    # Case: flat (images, hints, *extras)
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return first, second

    # Rare fallback: (something, (images, hints), *extras)
    if second is not None:
        pair2 = _maybe_split_batched_pair(second)
        if pair2 is not None:
            images, hinted = pair2
            return _as_tensor(images), _as_tensor(hinted)

    raise ValueError(
        "Could not infer (images, hints) from train batch. "
        f"Types: first={type(first)}, second={type(second)}"
    )


def _unpack_batch_val(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expected val batch content:
      images: [B, 3, H, W]
      bool_hints_batch: [B, num_levels, Npatches]  (multiple hint levels)

    Supports:
      - dict with keys: 'image'/'images' and 'bool_hints_batch'/'bool_hinted_pos_all'/'hints_all'
      - tuple/list: (images, hints_all, *extras)
      - nested: ((images, hints_all), extra)
    """
    if isinstance(batch, dict):
        images = batch.get('image', batch.get('images', None))
        hints_all = (batch.get('bool_hints_batch', None) or
                     batch.get('bool_hinted_pos_all', None) or
                     batch.get('hints_all', None))
        if images is None or hints_all is None:
            raise ValueError(f"Dict batch missing required keys. Found keys: {list(batch.keys())}")
        return _as_tensor(images), _as_tensor(hints_all)

    if not isinstance(batch, (list, tuple)) or len(batch) < 1:
        raise ValueError(f"Unexpected batch structure (val): {type(batch)}")

    first = batch[0]
    second = batch[1] if len(batch) > 1 else None

    # Nested ((images, hints_all), extra)
    pair = _maybe_split_batched_pair(first)
    if pair is not None:
        images, hints_all = pair
        return _as_tensor(images), _as_tensor(hints_all)

    # Flat (images, hints_all, *extras)
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return first, second

    # Fallback: (something, (images, hints_all), *extras)
    if second is not None:
        pair2 = _maybe_split_batched_pair(second)
        if pair2 is not None:
            images, hints_all = pair2
            return _as_tensor(images), _as_tensor(hints_all)

    raise ValueError(
        "Could not infer (images, hints_all) from val batch. "
        f"Types: first={type(first)}, second={type(second)}"
    )


def _first_step_debug(step: int, name: str, obj: Any):
    """One-time lightweight structure print to help diagnose batch shapes/types."""
    if step == 0:
        if isinstance(obj, torch.Tensor):
            print(f"[DEBUG] {name}: Tensor shape={tuple(obj.shape)}, dtype={obj.dtype}")
        elif isinstance(obj, (list, tuple)):
            t = type(obj[0]).__name__ if len(obj) > 0 else "EMPTY"
            s = getattr(obj[0], 'shape', None) if len(obj) > 0 else None
            print(f"[DEBUG] {name}: {type(obj).__name__} len={len(obj)} first_elem={t} shape={s}")
        elif isinstance(obj, dict):
            print(f"[DEBUG] {name}: dict keys={list(obj.keys())[:8]}...")
        else:
            print(f"[DEBUG] {name}: type={type(obj).__name__}")


# -----------------------------
# Training / Validation
# -----------------------------
def train_one_epoch(
    model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
    log_writer=None, lr_scheduler=None, start_steps: Optional[int] = None,
    lr_schedule_values=None, wd_schedule_values=None, exp_name: Optional[str] = None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = torch.nn.MSELoss(reduction='none')
    lambda_tpr = 2.5

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % 100 == 0 and exp_name:
            print(f"[DEBUG] {exp_name} | Epoch: {epoch}, Step: {step}")

        # --- unpack batch robustly (handles dict/tuple/list/nested) ---
        _first_step_debug(step, "raw_batch", batch)
        images, bool_hinted_pos = _unpack_batch_train(batch)
        _first_step_debug(step, "images_after_unpack", images)
        _first_step_debug(step, "bool_hinted_pos_after_unpack", bool_hinted_pos)

        # --- move to device ---
        images = images.to(device, non_blocking=True)
        bool_hinted_pos = bool_hinted_pos.to(device, non_blocking=True).to(torch.bool)

        # step index for LR/WD schedule
        it = start_steps + step if start_steps is not None else step

        # --- per-step LR/WD scheduling (learning-rate fix stays) ---
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Convert to LAB color space
        images_lab = rgb2lab(images)  # [B, 3, H, W]
        B, C, H, W = images_lab.shape
        h, w = H // patch_size, W // patch_size

        # Prepare hint mask upsampled to image size
        side = int(math.sqrt(bool_hinted_pos.shape[1]))
        mask_2d = bool_hinted_pos.view(B, 1, side, side).float()
        full_mask_for_input = F.interpolate(mask_2d, size=(H, W), mode='nearest')
        full_mask_for_ab_channels = (full_mask_for_input > 0.5)

        # Mask ab channels outside hints
        images_lab_masked = images_lab.clone()
        images_lab_masked[:, 1, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
        images_lab_masked[:, 2, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)

        # Build 4-ch input (L,a_masked,b_masked, hint_mask)
        input_4ch_tensor = torch.cat((images_lab_masked, full_mask_for_input), dim=1)  # [B,4,H,W]

        # Prepare patch labels (ground-truth in LAB, we will take ab later)
        with torch.no_grad():
            images_patch = rearrange(
                images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size
            )
            labels = rearrange(
                images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size
            )  # [B, N, P, 3]

        # Forward + Loss
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_4ch_tensor, bool_hinted_pos)  # [B, N, P*2]
            outputs = rearrange(
                outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size, c=2
            )  # [B, N, P, 2] -> ab preds

            gt_ab = labels[:, :, :, 1:]  # take ab from LAB labels
            mse_pixelwise = loss_func(outputs, gt_ab)

            # Weight loss to emphasize hinted regions (TPR-oriented)
            gt_mask = (gt_ab.abs().sum(dim=-1) > 0).float().unsqueeze(-1)  # [B,N,P,1]
            weight = gt_mask * lambda_tpr + (1.0 - gt_mask)
            weighted_loss = mse_pixelwise * weight
            loss = weighted_loss.mean()

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # ---- logging (lr, min_lr, loss, grad_norm) ----
        current_lr = optimizer.param_groups[0]["lr"]
        current_min_lr = min([pg["lr"] for pg in optimizer.param_groups])
        metric_logger.update(lr=current_lr, min_lr=current_min_lr, loss=loss_value)

        optimizer.zero_grad(set_to_none=True)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=max_norm,
            parameters=model.parameters(), create_graph=is_second_order
        )
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm if isinstance(grad_norm, float) else float(grad_norm.item()))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module, data_loader: Iterable, device: torch.device,
    patch_size: int = 16, log_writer=None, val_hint_list=[10]
):
    model.eval()
    header = 'Validation'
    psnr_sum = {val_num: 0. for val_num in val_hint_list}
    num_validated = 0

    for step, (batch, _) in tqdm(enumerate(data_loader), desc=header, ncols=100, total=len(data_loader)):
        _first_step_debug(step, "raw_batch_val", batch)
        images, bool_hints_batch = _unpack_batch_val(batch)
        _first_step_debug(step, "images_val_after_unpack", images)
        _first_step_debug(step, "bool_hints_batch_after_unpack", bool_hints_batch)

        images = images.to(device, non_blocking=True)
        bool_hints_batch = bool_hints_batch.to(device, non_blocking=True).to(torch.bool)
        if bool_hints_batch.dim() == 2:  # [B, 196] -> [B, 1, 196]
            bool_hints_batch = bool_hints_batch.unsqueeze(1)

        # When you compute 'side', use the last dim:
        side = int(math.sqrt(bool_hints_batch.shape[-1]))  # 14

        B, _, H, W = images.shape
        h, w = H // patch_size, W // patch_size

        images_lab = rgb2lab(images)
        images_patch = rearrange(
            images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size
        )
        labels = rearrange(
            images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size
        )  # [B,N,P,3]

        for idx, count in enumerate(val_hint_list):
            bool_hint = bool_hints_batch[:, idx]  # [B, Npatches]
            side = int(math.sqrt(bool_hint.shape[1]))
            mask_2d = bool_hint.view(B, 1, side, side).float()
            full_mask_for_input = F.interpolate(mask_2d, size=(H, W), mode='nearest')

            val_images_lab_masked = images_lab.clone()
            full_mask_for_ab_channels = (full_mask_for_input > 0.5)
            val_images_lab_masked[:, 1, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)
            val_images_lab_masked[:, 2, :, :].masked_fill_(~full_mask_for_ab_channels.squeeze(1), 0)

            val_input_4ch_tensor = torch.cat((val_images_lab_masked, full_mask_for_input), dim=1)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(val_input_4ch_tensor, bool_hint)  # [B, N, P*2]
                outputs = rearrange(
                    outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size, p2=patch_size, c=2
                )  # -> [B,N,P,2]

            # Reconstruct predicted LAB (L from labels, ab from outputs)
            pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)  # [B,N,P,3]
            pred_imgs_lab = rearrange(
                pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                h=h, w=w, p1=patch_size, p2=patch_size
            )
            pred_imgs = lab2rgb(pred_imgs_lab)

            _psnr = psnr(images, pred_imgs) * B
            psnr_sum[count] += float(_psnr.item())

        num_validated += B

    psnr_avg = {f'psnr@{count}': psnr_sum[count] / max(1, num_validated) for count in val_hint_list}

    torch.cuda.synchronize()

    if log_writer is not None:
        log_writer.update(head="psnr", **psnr_avg)

    return psnr_avg
