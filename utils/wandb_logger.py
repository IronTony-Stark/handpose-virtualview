import wandb

enabled = True

run = wandb.init(
    project="Pose Estimation using OCT ICRA 2025",
    name=f"ICVL Multi-View",
    tags=["ICVL"],
    mode="online" if enabled else "disabled",
)


class SummaryWriter:
    def __init__(self, log_dir, phase):
        self.phase = phase

    def add_scalar(self, metric, value, global_step):
        wandb.log({f"{self.phase}/{metric}": value}, step=global_step)

    def add_images(self, metric, images, global_step):
        wandb.log({f"{self.phase}/{metric}": [wandb.Image(image) for image in images]}, step=global_step)
