from dataclasses import dataclass

import torch
import wandb
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler

from scripts.utils import KL_loss, define_instance, dynamic_infer
from VAE_finetune_utils.logging_utils import log_validation_visuals_to_wandb, pick_visualization_indices


@dataclass
class TrainState:
    best_val_loss: float = 1e12
    global_step: int = 0


class VAEFinetuneTrainer:
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self.state = TrainState()

        args.autoencoder_def["num_splits"] = 1
        self.autoencoder = define_instance(args, "autoencoder_def").to(device)
        self.discriminator = PatchDiscriminator(
            spatial_dims=args.spatial_dims,
            num_layers_d=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            norm="INSTANCE",
        ).to(device)

        if args.recon_loss == "l2":
            self.intensity_loss = MSELoss()
        else:
            self.intensity_loss = L1Loss(reduction="mean")

        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        self.perceptual_loss = None
        if not getattr(args, "disable_perceptual_loss", False):
            self.perceptual_loss = PerceptualLoss(
                spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
            ).eval().to(device)

        self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)

        self.optimizer_g = torch.optim.Adam(params=self.autoencoder.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)

        self.scheduler_g = lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=self._warmup_rule)
        self.scheduler_d = lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=self._warmup_rule)

        self.scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5, enabled=args.amp)
        self.scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5, enabled=args.amp)

        self.val_inferer = (
            SlidingWindowInferer(
                roi_size=args.val_sliding_window_patch_size,
                sw_batch_size=1,
                progress=False,
                overlap=0.0,
                device=torch.device("cpu"),
                sw_device=device,
            )
            if args.val_sliding_window_patch_size
            else SimpleInferer()
        )

    def _warmup_rule(self, epoch: int) -> float:
        if epoch < self.args.lr_warmup_epoch1:
            return self.args.lr_warmup_scale1
        if epoch < self.args.lr_warmup_epoch2:
            return self.args.lr_warmup_scale2
        return 1.0

    def loss_weighted_sum(self, losses: dict[str, float | torch.Tensor]):
        return losses["recons_loss"] + self.args.kl_weight * losses["kl_loss"] + self.args.perceptual_weight * losses["p_loss"]

    def load_pretrained_if_needed(self):
        if not self.args.finetune:
            return
        checkpoint_autoencoder = torch.load(self.args.trained_autoencoder_path, map_location="cpu")
        if "unet_state_dict" in checkpoint_autoencoder:
            checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
        self.autoencoder.load_state_dict(checkpoint_autoencoder)


    def _compute_perceptual_loss(self, reconstruction: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if self.perceptual_loss is None:
            return torch.zeros((), dtype=reconstruction.dtype, device=reconstruction.device)
        return self.perceptual_loss(reconstruction.float(), images.float())

    def _infer_with_fallback(self, images: torch.Tensor):
        try:
            return dynamic_infer(self.val_inferer, self.autoencoder, images)
        except NotImplementedError as err:
            if "slow_conv3d_forward" not in str(err):
                raise
            if self.device.type != "cuda":
                raise
            autoencoder_cpu = self.autoencoder.to("cpu")
            images_cpu = images.float().cpu()
            reconstruction, z_mu, z_sigma = autoencoder_cpu(images_cpu)
            self.autoencoder.to(self.device)
            return reconstruction.to(self.device), z_mu.to(self.device), z_sigma.to(self.device)

    def train_one_epoch(self, dataloader_train, dataloader_val, epoch: int):
        self.autoencoder.train()
        self.discriminator.train()
        epoch_losses = {"recons_loss": 0.0, "kl_loss": 0.0, "p_loss": 0.0, "g_adv": 0.0, "d": 0.0}

        for batch in dataloader_train:
            images = batch["image"].to(self.device).contiguous()
            self.optimizer_g.zero_grad(set_to_none=True)
            self.optimizer_d.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.args.amp):
                reconstruction, z_mu, z_sigma = self.autoencoder(images)
                losses = {
                    "recons_loss": self.intensity_loss(reconstruction, images),
                    "kl_loss": KL_loss(z_mu, z_sigma),
                    "p_loss": self._compute_perceptual_loss(reconstruction, images),
                }
                logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = self.loss_weighted_sum(losses) + self.args.adv_weight * generator_loss

            self.scaler_g.scale(loss_g).backward()
            self.scaler_g.unscale_(self.optimizer_g)
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()

            with autocast("cuda", enabled=self.args.amp):
                logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = self.discriminator(images.contiguous().detach())[-1]
                loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5

            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()

            self.state.global_step += 1
            wandb.log(
                {
                    "train/recons_loss_iter": losses["recons_loss"].item(),
                    "train/kl_loss_iter": losses["kl_loss"].item(),
                    "train/p_loss_iter": losses["p_loss"].item(),
                    "train/g_adv_iter": generator_loss.item(),
                    "train/d_loss_iter": loss_d.item(),
                    "train/lr": self.optimizer_g.param_groups[0]["lr"],
                },
                step=self.state.global_step,
            )

            epoch_losses["recons_loss"] += losses["recons_loss"].item()
            epoch_losses["kl_loss"] += losses["kl_loss"].item()
            epoch_losses["p_loss"] += losses["p_loss"].item()
            epoch_losses["g_adv"] += generator_loss.item()
            epoch_losses["d"] += loss_d.item()

            if self.args.val_every_n_steps > 0 and self.state.global_step % self.args.val_every_n_steps == 0:
                self._validate_and_checkpoint(dataloader_val=dataloader_val, epoch=epoch)

        self.scheduler_g.step()
        self.scheduler_d.step()

        for key in epoch_losses:
            epoch_losses[key] /= max(len(dataloader_train), 1)

        epoch_total = self.loss_weighted_sum(epoch_losses) + self.args.adv_weight * epoch_losses["g_adv"]
        wandb.log({"train/epoch_total_loss": epoch_total, **{f"train/{k}_epoch": v for k, v in epoch_losses.items()}}, step=self.state.global_step)
        return epoch_losses

    @torch.no_grad()
    def validate(self, dataloader_val, epoch: int):
        self.autoencoder.eval()
        val_losses = {"recons_loss": 0.0, "kl_loss": 0.0, "p_loss": 0.0}

        cached_originals = []
        cached_recons = []
        cached_paths = []
        target_vis_ids = set(pick_visualization_indices(len(dataloader_val.dataset), self.args.val_vis_num_images, seed=epoch + 12345))

        sample_idx_base = 0
        for batch in dataloader_val:
            images_dev = batch["image"].to(self.device).contiguous()
            with autocast("cuda", enabled=self.args.amp):
                reconstruction, z_mu, z_sigma = self._infer_with_fallback(images_dev)

            val_losses["recons_loss"] += self.intensity_loss(reconstruction, images_dev).item()
            val_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
            val_losses["p_loss"] += self._compute_perceptual_loss(reconstruction, images_dev).item()

            bs = images_dev.shape[0]
            for i in range(bs):
                if sample_idx_base + i in target_vis_ids:
                    cached_originals.append(images_dev[i].cpu())
                    cached_recons.append(reconstruction[i].cpu())
                    cached_paths.append(batch["path"][i])
            sample_idx_base += bs

        for key in val_losses:
            val_losses[key] /= max(len(dataloader_val), 1)

        val_total = self.loss_weighted_sum(val_losses)
        scale_factor_sample = 1.0 / z_mu.flatten().std()

        wandb.log(
            {
                "val/total_loss": val_total,
                "val/recons_loss": val_losses["recons_loss"],
                "val/kl_loss": val_losses["kl_loss"],
                "val/p_loss": val_losses["p_loss"],
                "val/scale_factor": scale_factor_sample.item(),
                "epoch": epoch,
            },
            step=self.state.global_step,
        )

        log_validation_visuals_to_wandb(cached_originals, cached_recons, cached_paths, epoch=epoch, global_step=self.state.global_step)
        return val_total

    def _validate_and_checkpoint(self, dataloader_val, epoch: int):
        val_loss = self.validate(dataloader_val, epoch=epoch)
        ckpt_state = {
            "autoencoder": self.autoencoder.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "epoch": epoch,
            "global_step": self.state.global_step,
        }
        torch.save(ckpt_state, self.args.latest_checkpoint_path)

        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            torch.save(self.autoencoder.state_dict(), self.args.best_autoencoder_path)
