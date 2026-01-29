import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

import numpy as np
from Cad_VLM.config.macro import *
from CadSeqProc.utility.logger import CLGLogger
from Cad_VLM.models.draw2cad import SVG2CADTransformer
from Cad_VLM.models.loss import CELoss, SpaceAwareLoss
from Cad_VLM.models.metrics import AccuracyCalculator
from Cad_VLM.models.utils import print_with_separator
from Cad_VLM.dataprep.t2c_dataset_new import get_draw2cad_dataloaders

from loguru import logger
import torch
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from Cad_VLM.models.scheduler import GradualWarmupScheduler
import gc
import argparse
import yaml
import warnings
import logging.config

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

t2clogger = CLGLogger().configure_logger(verbose=True).logger


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def save_yaml_file(yaml_data, filename, output_dir):
    with open(os.path.join(output_dir, filename), "w+") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


def setup_distributed(rank, world_size):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        torch.cuda.set_device(rank)

    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)


def cleanup_distributed():
    dist.destroy_process_group()


def analyze_gradients(loss, model, scaler=None):
    loss_list = loss if isinstance(loss, list) else [loss]
    params = [p for p in model.parameters() if p.requires_grad]
    flats = []
    norms = []
    cos = []
    for loss in loss_list:
        if scaler is not None:
            loss = scaler.scale(loss)
        grads = torch.autograd.grad(loss, params,
                                     retain_graph=True, allow_unused=True)
        scale = scaler.get_scale() if scaler is not None else 1.0
        grads = [g.detach() / scale for g in grads if g is not None]
        flats.append(torch.cat([g.flatten() for g in grads]))
    
    norms = [flat.norm().item() for flat in flats]
    for i in range(len(loss_list) - 1):
        for j in range(i+1, len(loss_list)):
            cos.append(torch.dot(flats[i], flats[j]).item() / (norms[i] * norms[j] + eps))

    return flats, norms, cos


@logger.catch()
def main():
    print_with_separator("😊 Draw2CAD Training 😊")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/trainer.yaml",
    )
    args = parser.parse_args()
    config = parse_config_file(args.config_path)

    # Determine whether to use torchrun (env vars) or spawn
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeError("No GPUs available for training")

    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", ngpus))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        run(local_rank, world_size, config, args)
    else:
        world_size = ngpus
        mp.spawn(run, args=(world_size, config, args), nprocs=world_size, join=True)


def run(rank, world_size, config, args):
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() \
        else torch.device("cpu")

    t2clogger.info("Process {rank}/{world_size} using device {device}", rank=rank, world_size=world_size, device=device)

    # --------------------- Load Model ------------------------ #
    cad_config = config["draw2cad"]
    cad_config["device"] = device

    model = SVG2CADTransformer.from_config(cad_config).to(device)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "bn" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": 0.},
        {"params": no_decay_params, "weight_decay": 0.},
    ], lr=config["train"]["lr"])

    scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=1.0,
        total_epoch=config["train"]["warm_up"]
    )
    criterions = [
        CELoss(device=device),
        SpaceAwareLoss(device=device).to(device)
    ]
    scaler = GradScaler()

    log_dir = os.path.join(
        config["train"]["log_dir"]
    )

    if not config["debug"] and rank == 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_yaml_file(config, filename=args.config_path.split("/")[-1], output_dir=log_dir)

    # Start training
    train_model(
        model=model,
        criterions=criterions,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        log_dir=log_dir,
        num_epochs=config["train"]["num_epochs"],
        checkpoint_name=f"initial",
        config=config,
        rank=rank,
        world_size=world_size,
    )

    cleanup_distributed()


def train_model(
    model,
    criterions,
    optimizer,
    scheduler,
    scaler,
    device,
    log_dir,
    num_epochs,
    checkpoint_name,
    config,
    rank=0,
    world_size=1,
):
    """
    Trains a deep learning model (DDP-aware).
    """

    is_main = rank == 0

    # Create the dataloader for train
    train_loader, val_loader = get_draw2cad_dataloaders(
        cad_seq_dir=config["train_data"]["cad_seq_dir"],
        svg_dir=config["train_data"]["svg_dir"],
        split_filepath=config["train_data"]["split_filepath"],
        subsets=["train", "validation"],
        input_option=config["draw2cad"]["input_option"],
        batch_size=config["train"]["batch_size"],
        pin_memory=True,
        num_workers=min(config["train"]["num_workers"], os.cpu_count()),
        prefetch_factor=config["train"]["prefetch_factor"],
    )

    tensorboard_dir = os.path.join(log_dir, f"summary")

    # ---------------------- Resume Training from checkpoint --------------------- #
    checkpoint_file = os.path.join(log_dir, f"{checkpoint_name}.pth")

    if config["train"]["checkpoint_path"] is None:
        old_checkpoint_file = checkpoint_file
    else:
        old_checkpoint_file = config["train"]["checkpoint_path"]

    start_epoch = 1
    step = 0
    if os.path.exists(old_checkpoint_file):
        t2clogger.info("Using saved checkpoint at {}", old_checkpoint_file)
        checkpoint = torch.load(old_checkpoint_file, map_location=device)
        # model is DDP-wrapped; load to model.module
        try:
            model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        except Exception:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        step = checkpoint.get("step", 0)

    t2clogger.info("Saving checkpoint at {}", checkpoint_file)

    # Create the tensorboard summary writer only on main process
    writer = None
    if is_main:
        writer = SummaryWriter(log_dir=tensorboard_dir, comment=f"{checkpoint_name}")

    # ---------------------------------- Training ---------------------------------- #
    # model is DDP instance
    model.train()
    for epoch in range(start_epoch, num_epochs + 1):
        # set epoch for sampler
        train_loader.sampler.set_epoch(epoch)

        train_ce_loss = []
        train_skt_loss = []
        train_ext_loss = []
        train_accuracy = []
        val_accuracy = []

        # Use tqdm on main process; other ranks get a simple loop
        if is_main:
            data_iterator = tqdm(train_loader, ascii=True, desc=f"[{epoch}/{num_epochs+1}]")
        else:
            data_iterator = train_loader

        for uids, vec_dict, mask_cad_dict, svg_dict in data_iterator:
            step += 1

            for key, value in vec_dict.items():
                vec_dict[key] = value.to(device)

            for key, value in mask_cad_dict.items():
                mask_cad_dict[key] = value.to(device)
            
            for key, value in svg_dict.items():
                svg_dict[key] = value.to(device)

            # Forward
            with autocast():
                cad_vec_pred, _ = model(
                    vec_dict=vec_dict,
                    mask_cad_dict=mask_cad_dict,
                    svg_dict=svg_dict
                ) # (B,N1,2,C1)

                # Loss
                ce_loss, loss_sep_dict = criterions[0](
                    {
                        "pred": cad_vec_pred,
                        "target": vec_dict["cad_vec_target"],
                        "key_padding_mask": ~mask_cad_dict["shifted_key_padding_mask"]
                    }
                )

                if SHAPE_LOSS:
                    skt_loss, ext_loss, space_loss_dict = criterions[1](
                        pred=cad_vec_pred, vec_dict=vec_dict, mask_cad_dict=mask_cad_dict)
                else:
                    skt_loss = torch.tensor(0., device=device)
                    ext_loss = torch.tensor(0., device=device)
                    space_loss_dict = {
                        "sketch_loss": 100 * skt_loss.item(),
                        "extrusion_loss": 100 * ext_loss.item(),
                        "valid_sketch": 0,
                        "valid_extrusion": 0,
                    }

            skt_loss = 10 * skt_loss
            ext_loss = 10 * ext_loss

            if SHAPE_LOSS:
                if is_main and step % 100 == 0:
                    _, norms, cos = analyze_gradients(
                        [ce_loss, skt_loss, ext_loss], model=model, scaler=scaler)
                    print(f"norms:{norms}\ncosine:{cos}\n")

            loss = ce_loss + skt_loss + ext_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            # Logging
            if is_main:
                train_ce_loss.append(loss_sep_dict["loss_seq"])
                train_skt_loss.append(space_loss_dict["sketch_loss"])
                train_ext_loss.append(space_loss_dict["extrusion_loss"])

                cad_accuracy = AccuracyCalculator(discard_token=len(END_TOKEN)).calculateAccMulti2DFromProbability(cad_vec_pred, vec_dict["cad_vec_target"])
                train_accuracy.append(cad_accuracy)

                updated_dict = {
                    "lr": optimizer.param_groups[0]["lr"],
                    "ce": np.round(train_ce_loss[-1], decimals=2),
                    "skt": np.round(train_skt_loss[-1], decimals=2),
                    "ext": np.round(train_ext_loss[-1], decimals=2),
                    "acc": np.round(train_accuracy[-1], decimals=2),
                }
                # Update the progress bar if present
                if isinstance(data_iterator, tqdm):
                    data_iterator.set_postfix(updated_dict)

                # Tensorboard
                if writer is not None and not config["debug"]:
                    writer.add_scalar("CE Loss", np.mean(train_ce_loss), step,new_style=False)
                    writer.add_scalar("SKT Loss", np.mean(train_skt_loss), step, new_style=False)
                    writer.add_scalar("EXT Loss", np.mean(train_ext_loss), step, new_style=False)
                    writer.add_scalar("Train Acc", np.mean(train_accuracy), step, new_style=False)

        if epoch % config["val"]["frequency"] == 0:
            val_cad_acc = validation_one_epoch(
                val_loader=val_loader,
                model=model,
                device=device,
                config=config,
                is_main=is_main,
                epoch=epoch,
                num_epochs=num_epochs,
                writer=writer,
                topk=1,
            )
            val_accuracy.append(val_cad_acc)

        # Save checkpoints only on main process
        if not config["debug"] and is_main:
            if epoch % config["train"]["checkpoint_interval"] == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.get_trainable_state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "step": step,
                    },
                    os.path.join(log_dir, f"epoch_{epoch}.pth"),
                )

        # Print epoch summary from main
        if is_main and epoch % config["val"]["frequency"] == 0:
            logger.info(
                f"Epoch [{epoch}/{num_epochs+1}]✅,"
                f" Train Acc: {np.round(np.mean(train_accuracy), decimals=2)},"
                f" Val Acc: {np.round(np.mean(val_accuracy), decimals=2)}",
            )

    if is_main and writer is not None:
        writer.close()
    t2clogger.success("Training Finished.")

@torch.no_grad()
def validation_one_epoch(
    val_loader,
    model,
    device,
    config,
    is_main=False,
    epoch=0,
    num_epochs=0,
    writer=None,
    topk=5,
):
    seq_acc_all = []

    # model might be DDP; use module for actual model
    val_model = model.module if hasattr(model, "module") else model
    val_model.eval()

    if is_main:
        data_iterator = tqdm(val_loader, ascii=True, desc=f"Epoch [{epoch}/{num_epochs+1}] Validation✨")
    else:
        data_iterator = val_loader

    total_batch, cur_batch = config["val"]["val_batch"], 0

    for uids, vec_dict, mask_cad_dict, svg_dict in data_iterator:
        if cur_batch == total_batch:
            break
        cur_batch += 1

        for key, value in vec_dict.items():
                vec_dict[key] = value.to(device)

        for key, value in mask_cad_dict.items():
            mask_cad_dict[key] = value.to(device)
        
        for key, value in svg_dict.items():
            svg_dict[key] = value.to(device)

        sec_topk_acc = []
        for topk_index in range(1, topk + 1):
            with autocast():
                pred_cad_seq_dict = val_model.test_decode(
                    svg_dict=svg_dict,
                    maxlen=MAX_CAD_SEQUENCE_LENGTH,
                    nucleus_prob=0,
                    topk_index=topk_index,
                    device=device,
                )

            gc.collect()
            torch.cuda.empty_cache()
            try:
                cad_seq_acc = AccuracyCalculator(discard_token=len(END_TOKEN)).calculateAccMulti2DFromLabel(pred_cad_seq_dict["cad_vec"].cpu(), vec_dict["cad_vec"].cpu())
            except Exception as e:
                logger.error(f"Error: {e}")
                cad_seq_acc = 0

            if isinstance(data_iterator, tqdm):
                data_iterator.set_postfix({"Val Acc": np.round(cad_seq_acc, decimals=2)})

            sec_topk_acc.append(cad_seq_acc)

        seq_acc_all.append(np.max(sec_topk_acc))

    mean_seq_acc = np.mean(seq_acc_all) if len(seq_acc_all) > 0 else 0

    gc.collect()
    torch.cuda.empty_cache()

    if writer is not None:
        writer.add_scalar("Val Acc", np.round(mean_seq_acc, decimals=2), epoch, new_style=False)

    return mean_seq_acc


if __name__ == "__main__":
    main()