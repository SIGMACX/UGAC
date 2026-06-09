from __future__ import annotations

import argparse
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
try:
    from torch.amp import GradScaler, autocast

    _TORCH_AMP_HAS_DEVICE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    _TORCH_AMP_HAS_DEVICE = False
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            self.enabled = False

        def add_scalar(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None

from mmwhs_fns.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from mmwhs_fns.datasets import MMWHSDataset
from mmwhs_fns.fusion import UncertaintyGuidedAdaptiveFuser
from mmwhs_fns.losses import (
    class_weights_from_dataset,
    consistency_weight,
    dice_loss,
    discriminator_adversarial_loss,
    generator_adversarial_loss,
    masked_mse_loss,
    mse_consistency_loss,
)
from mmwhs_fns.metrics import compute_confusion_counts, dice_from_counts
from mmwhs_fns.models import build_model
from mmwhs_fns.models.critic import PanDiscriminator


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def amp_autocast(enabled: bool):
    if _TORCH_AMP_HAS_DEVICE:
        return autocast("cuda", enabled=enabled)
    return autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    if _TORCH_AMP_HAS_DEVICE:
        return GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args()

    defaults: dict[str, Any] = {}
    if known.config is not None:
        defaults = yaml.safe_load(known.config.read_text(encoding="utf-8")) or {}

    parser = argparse.ArgumentParser(description="Train the MMWHS FNS experiment.")
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument("--data_root", type=Path, default=Path("data/MMWHS"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/mmwhs"))
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_ratio", type=int, default=20)
    parser.add_argument("--model_name_1", type=str, default="mamba_unet")
    parser.add_argument("--model_name_2", type=str, default="unet")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--seg_lr", type=float, default=1e-4)
    parser.add_argument("--adv_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seg_max_norm", type=float, default=5.0)
    parser.add_argument("--adv_max_norm", type=float, default=1.0)
    parser.add_argument("--lambda_sup", type=float, default=1.0)
    parser.add_argument("--mu_adv", type=float, default=0.01)
    parser.add_argument("--consistency", type=float, default=0.4)
    parser.add_argument("--consistency_rampup", type=float, default=100.0)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    parser.add_argument("--fuse_method", type=str, default="uncertainty_fuse")
    parser.add_argument("--fuse_weights", type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument("--evaluation_interval", type=int, default=1)
    parser.add_argument("--max_no_improve_epochs", type=int, default=200)
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--labeled_txt", type=Path, default=None)
    parser.add_argument("--unlabeled_txt", type=Path, default=None)
    parser.add_argument("--test_txt", type=Path, default=None)
    parser.add_argument("--label_values", type=int, nargs="+", default=[0, 60, 126, 150, 165, 179, 246, 255])
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    for field in ("data_root", "output_dir", "resume", "labeled_txt", "unlabeled_txt", "test_txt"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, Path(value))
    return args


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging(output_dir: Path, rank: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = []
    if rank == 0:
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def build_split_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    txt_root = args.data_root / "txt_path"
    labeled = args.labeled_txt or txt_root / f"train_imagesTr_labeled_{args.label_ratio}.txt"
    unlabeled = args.unlabeled_txt or txt_root / f"train_imagesTr_unlabeled_{100 - args.label_ratio}.txt"
    test = args.test_txt or txt_root / "test_imagesTs.txt"
    return Path(labeled), Path(unlabeled), Path(test)


def make_loader(dataset, batch_size: int, num_workers: int, distributed: bool, rank: int, world_size: int, shuffle: bool):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def balanced_loader(loader, total_batches: int):
    iterator = iter(loader)
    for _ in range(total_batches):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)


def first_output(model_output):
    return model_output[0] if isinstance(model_output, (tuple, list)) else model_output


def maybe_sync_batchnorm(model: nn.Module, distributed: bool) -> nn.Module:
    if distributed and torch.cuda.is_available():
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def wrap_ddp(model: nn.Module, distributed: bool, local_rank: int) -> nn.Module:
    if not distributed:
        return model
    device_ids = [local_rank] if torch.cuda.is_available() else None
    return DDP(model, device_ids=device_ids)


@torch.no_grad()
def validate(model, loader, args, device, distributed: bool, rank: int, output_dir: Path) -> float:
    model.eval()
    total_tp = torch.zeros(args.num_classes, dtype=torch.float64, device=device)
    total_fp = torch.zeros(args.num_classes, dtype=torch.float64, device=device)
    total_fn = torch.zeros(args.num_classes, dtype=torch.float64, device=device)

    iterable = tqdm(loader, desc="Validation", disable=rank != 0)
    for images, labels, _, _ in iterable:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = first_output(model(images))
        pred = logits.argmax(dim=1)

        tp, fp, fn, _ = compute_confusion_counts(pred.cpu().numpy(), labels.cpu().numpy(), args.num_classes)
        total_tp += torch.as_tensor(tp, dtype=torch.float64, device=device)
        total_fp += torch.as_tensor(fp, dtype=torch.float64, device=device)
        total_fn += torch.as_tensor(fn, dtype=torch.float64, device=device)

    if distributed:
        dist.all_reduce(total_tp, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_fp, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_fn, op=dist.ReduceOp.SUM)

    dice_per_class = dice_from_counts(total_tp, total_fp, total_fn)
    return dice_per_class[1:].mean().item()


def train_one_epoch(
    model1,
    model2,
    critic1,
    critic2,
    train_loader,
    unlabel_loader,
    optimizer1,
    optimizer2,
    critic_optimizer1,
    critic_optimizer2,
    ce_loss,
    fuser,
    scalers,
    args,
    device,
    epoch: int,
    rank: int,
    writer,
) -> None:
    model1.train()
    model2.train()
    critic1.train()
    critic2.train()
    max_batches = max(len(train_loader), len(unlabel_loader))
    labeled_iter = balanced_loader(train_loader, max_batches)
    unlabeled_iter = balanced_loader(unlabel_loader, max_batches)
    amp_enabled = bool(args.use_amp and torch.cuda.is_available())

    progress = tqdm(zip(labeled_iter, unlabeled_iter), total=max_batches, desc=f"Epoch {epoch + 1}/{args.num_epochs}", disable=rank != 0)
    for step, (batch_labeled, batch_unlabeled) in enumerate(progress):
        images_l, labels_l, _, _ = batch_labeled
        images_u, _, _, _ = batch_unlabeled
        images_l = images_l.to(device, non_blocking=True)
        labels_l = labels_l.to(device, non_blocking=True)
        images_u = images_u.to(device, non_blocking=True)

        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)

        with amp_autocast(enabled=amp_enabled):
            logits1_l = first_output(model1(images_l))
            logits1_u = first_output(model1(images_u))
            logits2_l = first_output(model2(images_l))
            logits2_u = first_output(model2(images_u))

            fused_u = fuser.fuse(logits1_u, logits2_u)
            current_weight = consistency_weight(epoch * max_batches + step, args.consistency, args.consistency_rampup)

            sup1 = args.lambda_sup * (ce_loss(logits1_l, labels_l) + dice_loss(logits1_l, labels_l, args.num_classes))
            sup2 = args.lambda_sup * (ce_loss(logits2_l, labels_l) + dice_loss(logits2_l, labels_l, args.num_classes))
            diff1 = mse_consistency_loss(logits1_u, fused_u)
            diff2 = mse_consistency_loss(logits2_u, fused_u)

            prob1_l = F.softmax(logits1_l, dim=1)
            prob2_l = F.softmax(logits2_l, dim=1)
            prob_fused_u = F.softmax(fused_u, dim=1)

            for param in critic1.parameters():
                param.requires_grad_(False)
            for param in critic2.parameters():
                param.requires_grad_(False)

            adv_gen1 = generator_adversarial_loss(critic1(prob1_l))
            adv_gen2 = generator_adversarial_loss(critic2(prob2_l))
            conf1 = torch.sigmoid(critic2(prob_fused_u))
            conf2 = torch.sigmoid(critic1(prob_fused_u))
            unsup1 = diff1 + masked_mse_loss(logits1_u, fused_u, conf1, args.mask_threshold)
            unsup2 = diff2 + masked_mse_loss(logits2_u, fused_u, conf2, args.mask_threshold)
            gen1 = sup1 + current_weight * unsup1 + args.mu_adv * adv_gen1
            gen2 = sup2 + current_weight * unsup2 + args.mu_adv * adv_gen2

        scalers["gen1"].scale(gen1).backward(retain_graph=True)
        scalers["gen1"].unscale_(optimizer1)
        torch.nn.utils.clip_grad_norm_(model1.parameters(), args.seg_max_norm)
        scalers["gen1"].step(optimizer1)
        scalers["gen1"].update()

        scalers["gen2"].scale(gen2).backward()
        scalers["gen2"].unscale_(optimizer2)
        torch.nn.utils.clip_grad_norm_(model2.parameters(), args.seg_max_norm)
        scalers["gen2"].step(optimizer2)
        scalers["gen2"].update()

        for param in critic1.parameters():
            param.requires_grad_(True)
        for param in critic2.parameters():
            param.requires_grad_(True)

        one_hot = F.one_hot(labels_l, num_classes=args.num_classes).permute(0, 3, 1, 2).float()
        with amp_autocast(enabled=amp_enabled):
            fake1 = critic1(prob1_l.detach())
            real1 = critic1(one_hot)
            fake2 = critic2(prob2_l.detach())
            real2 = critic2(one_hot)
            adv1 = discriminator_adversarial_loss(fake1, real1)
            adv2 = discriminator_adversarial_loss(fake2, real2)

        critic_optimizer1.zero_grad(set_to_none=True)
        scalers["critic1"].scale(adv1).backward()
        scalers["critic1"].unscale_(critic_optimizer1)
        torch.nn.utils.clip_grad_norm_(critic1.parameters(), args.adv_max_norm)
        scalers["critic1"].step(critic_optimizer1)
        scalers["critic1"].update()

        critic_optimizer2.zero_grad(set_to_none=True)
        scalers["critic2"].scale(adv2).backward()
        scalers["critic2"].unscale_(critic_optimizer2)
        torch.nn.utils.clip_grad_norm_(critic2.parameters(), args.adv_max_norm)
        scalers["critic2"].step(critic_optimizer2)
        scalers["critic2"].update()

        if rank == 0:
            global_step = epoch * max_batches + step
            writer.add_scalar("train/gen1", gen1.item(), global_step)
            writer.add_scalar("train/gen2", gen2.item(), global_step)
            writer.add_scalar("train/sup1", sup1.item(), global_step)
            writer.add_scalar("train/sup2", sup2.item(), global_step)
            writer.add_scalar("train/unsup1", unsup1.item(), global_step)
            writer.add_scalar("train/unsup2", unsup2.item(), global_step)
            writer.add_scalar("train/critic1", adv1.item(), global_step)
            writer.add_scalar("train/critic2", adv2.item(), global_step)
            progress.set_postfix({"gen1": f"{gen1.item():.3f}", "gen2": f"{gen2.item():.3f}"})


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / f"{run_name}_{args.model_name_1}_{args.model_name_2}_label{args.label_ratio}"
    configure_logging(output_dir, rank)

    if rank == 0:
        logging.info("Arguments: %s", vars(args))
        logging.info("World size: %s", world_size)

    labeled_txt, unlabeled_txt, test_txt = build_split_paths(args)
    train_dataset = MMWHSDataset(labeled_txt, args.data_root / "imagesTr", args.data_root / "labelsTr", args.image_size, args.num_classes, args.label_values)
    unlabel_dataset = MMWHSDataset(unlabeled_txt, args.data_root / "imagesTr", None, args.image_size, args.num_classes, args.label_values)
    test_dataset = MMWHSDataset(test_txt, args.data_root / "imagesTr", args.data_root / "labelsTr", args.image_size, args.num_classes, args.label_values)

    train_loader, train_sampler = make_loader(train_dataset, args.batch_size, args.num_workers, distributed, rank, world_size, True)
    unlabel_loader, unlabel_sampler = make_loader(unlabel_dataset, args.batch_size, args.num_workers, distributed, rank, world_size, True)
    test_loader, _ = make_loader(test_dataset, args.batch_size * 2, args.num_workers, distributed, rank, world_size, False)

    model1 = build_model(args.model_name_1, args.num_classes, args.input_channels, args.image_size, args.patch_size).to(device)
    model2 = build_model(args.model_name_2, args.num_classes, args.input_channels, args.image_size, args.patch_size).to(device)
    critic1 = PanDiscriminator(args.num_classes).to(device)
    critic2 = PanDiscriminator(args.num_classes).to(device)

    model1 = wrap_ddp(maybe_sync_batchnorm(model1, distributed), distributed, local_rank)
    model2 = wrap_ddp(maybe_sync_batchnorm(model2, distributed), distributed, local_rank)
    critic1 = wrap_ddp(maybe_sync_batchnorm(critic1, distributed), distributed, local_rank)
    critic2 = wrap_ddp(maybe_sync_batchnorm(critic2, distributed), distributed, local_rank)

    counts, weights = class_weights_from_dataset(train_dataset, args.num_classes, device, distributed)
    if rank == 0:
        logging.info("Class counts: %s", counts.cpu().tolist())
        logging.info("Class weights: %s", weights.cpu().tolist())

    ce_loss = nn.CrossEntropyLoss(weight=weights)
    optimizer1 = optim.AdamW(model1.parameters(), lr=args.seg_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    optimizer2 = optim.AdamW(model2.parameters(), lr=args.seg_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    critic_optimizer1 = optim.AdamW(critic1.parameters(), lr=args.adv_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    critic_optimizer2 = optim.AdamW(critic2.parameters(), lr=args.adv_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.num_epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.num_epochs)
    critic_scheduler1 = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer1, T_max=args.num_epochs)
    critic_scheduler2 = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer2, T_max=args.num_epochs)

    fuser = UncertaintyGuidedAdaptiveFuser(args.fuse_method, args.fuse_weights, num_classes=args.num_classes).to(device)
    amp_enabled = bool(args.use_amp and torch.cuda.is_available())
    scalers = {
        "gen1": make_grad_scaler(enabled=amp_enabled),
        "gen2": make_grad_scaler(enabled=amp_enabled),
        "critic1": make_grad_scaler(enabled=amp_enabled),
        "critic2": make_grad_scaler(enabled=amp_enabled),
    }

    checkpoint_path = args.resume or output_dir / "checkpoint.pth.tar"
    start_epoch, best_dice, no_improve_epoch = load_checkpoint(
        checkpoint_path,
        {"model1": model1, "model2": model2, "critic1": critic1, "critic2": critic2},
        {"optimizer1": optimizer1, "optimizer2": optimizer2, "critic_optimizer1": critic_optimizer1, "critic_optimizer2": critic_optimizer2},
        device,
    )

    writer = SummaryWriter(output_dir / "runs") if rank == 0 else None
    start_time = time.time()
    try:
        for epoch in range(start_epoch, args.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if unlabel_sampler is not None:
                unlabel_sampler.set_epoch(epoch)

            train_one_epoch(
                model1,
                model2,
                critic1,
                critic2,
                train_loader,
                unlabel_loader,
                optimizer1,
                optimizer2,
                critic_optimizer1,
                critic_optimizer2,
                ce_loss,
                fuser,
                scalers,
                args,
                device,
                epoch,
                rank,
                writer,
            )

            scheduler1.step()
            scheduler2.step()
            critic_scheduler1.step()
            critic_scheduler2.step()

            if (epoch + 1) % args.evaluation_interval == 0:
                mean_dice = validate(model1, test_loader, args, device, distributed, rank, output_dir)
                if rank == 0:
                    logging.info("Epoch %d mean foreground Dice: %.4f", epoch + 1, mean_dice)
                    writer.add_scalar("validation/mean_dice", mean_dice, epoch + 1)

                    checkpoint_state = {
                        "epoch": epoch + 1,
                        "state_dict_model1": unwrap_model(model1).state_dict(),
                        "state_dict_model2": unwrap_model(model2).state_dict(),
                        "state_dict_critic1": unwrap_model(critic1).state_dict(),
                        "state_dict_critic2": unwrap_model(critic2).state_dict(),
                        "optimizer1": optimizer1.state_dict(),
                        "optimizer2": optimizer2.state_dict(),
                        "critic_optimizer1": critic_optimizer1.state_dict(),
                        "critic_optimizer2": critic_optimizer2.state_dict(),
                        "best_dice": best_dice,
                        "no_improve_epoch": no_improve_epoch,
                    }
                    save_checkpoint(output_dir / "checkpoint.pth.tar", checkpoint_state)

                    if mean_dice > best_dice:
                        best_dice = mean_dice
                        no_improve_epoch = 0
                        torch.save(unwrap_model(model1).state_dict(), output_dir / "best_model1.pth")
                        logging.info("Saved new best model with Dice %.4f", best_dice)
                    else:
                        no_improve_epoch += 1

                    if no_improve_epoch >= args.max_no_improve_epochs:
                        logging.info("Early stopping after %d epochs without improvement.", no_improve_epoch)
                        break
    finally:
        if rank == 0:
            elapsed = time.time() - start_time
            logging.info("Finished. Best Dice: %.4f. Elapsed seconds: %.1f", best_dice, elapsed)
            if writer is not None:
                writer.close()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
