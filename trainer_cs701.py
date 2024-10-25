import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


def trainer_cs701(args, model, snapshot_path):
    """
    Training function for cs701 segmentation model.

    Args:
        args: Namespace object containing training parameters including:
            - base_lr (float): Base learning rate
            - num_classes (int): Number of segmentation classes
            - batch_size (int): Training batch size
            - n_gpu (int): Number of GPUs to use
            - img_size (int): Size of input images
            - max_epochs (int): Maximum number of training epochs
            - eval_interval (int): Epochs between evaluations
        model: Neural network model to train
        snapshot_path (str): Directory to save model checkpoints and logs

    Returns:
        str: "Training Finished!" upon completion
    """
    # Import dataset classes
    from datasets.dataset_cs701 import Cs701_dataset, RandomGenerator

    # Configure args if not provided
    if not hasattr(args, "num_workers"):
        args.num_workers = 4
    if not hasattr(args, "eval_interval"):
        args.eval_interval = 5

    # Set up logging
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Initialize training parameters
    base_lr = args.base_lr  # Base learning rate
    num_classes = args.num_classes  # Output channel of network
    batch_size = args.batch_size * args.n_gpu  # Batch size per GPU

    train_path = os.path.join(args.root_path, "train_npz")
    val_path = os.path.join(args.root_path, "val_npz")

    # Create training and validation datasets
    db_train = Cs701_dataset(
        base_dir=train_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        ),
    )

    db_val = Cs701_dataset(
        base_dir=val_path,
        list_dir=args.list_dir,
        split="val",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        ),
    )
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    # Initialize data loaders with worker seed for reproducibility
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        db_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # Setup model for training
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # Initialize loss functions
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # Initialize optimizer with SGD (TODO: Consider changing to Adam)
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    # Initialize tensorboard writer for logging
    writer = SummaryWriter(snapshot_path + "/log")

    # Training loop setup
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info(
        "{} iterations per epoch. {} max iterations ".format(
            len(train_loader), max_iterations
        )
    )

    # Main training loop
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0

        # Training epoch
        for i_batch, sampled_batch in tqdm(
            enumerate(train_loader),
            desc=f"Train: {epoch_num}",
            total=len(train_loader),
            leave=False,
        ):
            # Get batch data
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass
            outputs = model(image_batch)

            # Calculate losses
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate with polynomial decay
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1

            # Log metrics
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/loss_ce", loss_ce, iter_num)

            batch_ce_loss += loss_ce.item()
            batch_dice_loss += loss_dice.item()

            # Visualization logging every 20 iterations
            if iter_num % 20 == 0:
                # Log example image, prediction, and ground truth
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image("train/Image", image, iter_num)
                outputs = torch.argmax(
                    torch.softmax(outputs, dim=1), dim=1, keepdim=True
                )
                writer.add_image("train/Prediction", outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)

        # Calculate average losses for the epoch
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info(
            "Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f"
            % (epoch_num, batch_loss, batch_ce_loss, batch_dice_loss)
        )

        # Validation phase
        if (epoch_num + 1) % args.eval_interval == 0:
            """
            Evaluate the model on validation set every eval_interval epochs.
            Saves the best model based on validation loss.
            """
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0

            # Validation loop
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(
                    enumerate(val_loader),
                    desc=f"Val: {epoch_num}",
                    total=len(val_loader),
                    leave=False,
                ):
                    # Get validation batch
                    image_batch, label_batch = (
                        sampled_batch["image"],
                        sampled_batch["label"],
                    )
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                    # Forward pass
                    outputs = model(image_batch)

                    # Calculate validation losses
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)

                    batch_ce_loss += loss_ce.item()
                    batch_dice_loss += loss_dice.item()

                # Calculate average validation losses
                batch_ce_loss /= len(val_loader)
                batch_dice_loss /= len(val_loader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss

                # Log validation results
                logging.info(
                    "Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f"
                    % (epoch_num, batch_loss, batch_ce_loss, batch_dice_loss)
                )

                # Save model checkpoints
                if batch_loss < best_loss:
                    # Save best model if current loss is better than previous best
                    save_mode_path = os.path.join(snapshot_path, "best_model.pth")
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = batch_loss
                else:
                    # Save last model regardless
                    save_mode_path = os.path.join(snapshot_path, "last_model.pth")
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    # Clean up and finish
    writer.close()
    return "Training Finished!"
