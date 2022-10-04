import torch
from pytorch_lightning.core.lightning import LightningModule
from hydra.utils import instantiate


from source.metric.MRRMetric import MRRMetric


class DescCodeModel(LightningModule):
    """Encodes the code and desc into an same space of embeddings."""

    def __init__(self, hparams):

        super(DescCodeModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.desc_encoder = instantiate(hparams.desc_encoder)
        self.code_encoder = instantiate(hparams.code_encoder)

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = MRRMetric(hparams.metric)

    def forward(self, desc, code):
        desc_repr = self.desc_encoder(desc)
        code_repr = self.code_encoder(code)
        return desc_repr, code_repr

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        desc_idx, desc, code_idx, code = batch["desc_idx"], batch["desc"], batch["code_idx"], batch["code"]
        desc_rpr, code_rpr = self.desc_encoder(desc), self.code_encoder(code)

        train_loss = self.loss(
            desc_idx,
            desc_rpr,
            code_idx,
            code_rpr
        )

        # log training loss
        self.log('train_LOSS', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        desc_idx, desc, code_idx, code = batch["desc_idx"], batch["desc"], batch["code_idx"], batch["code"]
        desc_rpr, code_rpr = self.desc_encoder(desc), self.code_encoder(code)

        self.mrr.update(
            desc_idx,
            desc_rpr,
            code_idx,
            code_rpr
        )

    def validation_epoch_end(self, outs):
        self.log("val_MRR", self.mrr.compute(), prog_bar=True)
        self.mrr.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, desc_idx, desc, code_idx, code = batch["idx"], batch["desc_idx"], batch["desc"], batch["code_idx"], batch["code"]
        desc_rpr, code_rpr = self.desc_encoder(desc), self.code_encoder(code)

        return {
            "idx": idx,
            "desc_idx": desc_idx,
            "desc_rpr": desc_rpr,
            "code_idx": code_idx,
            "code_rpr": code_rpr
        }

    def test_step(self, batch, batch_idx):
        desc, code = batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        self.log("test_MRR", self.mrr(desc_repr, code_repr), prog_bar=True)

    def test_epoch_end(self, outs):
        self.mrr.compute()

    def get_desc_encoder(self):
        return self.desc_encoder

    def get_code_encoder(self):
        return self.desc_encoder

    # def optimizer_step(
    #         self,
    #         epoch,
    #         batch_idx,
    #         optimizer,
    #         optimizer_idx,
    #         optimizer_closure,
    #         on_tpu=False,
    #         using_native_amp=False,
    #         using_lbfgs=False,
    # ):
    #     print(f"\n\nbatch_idx: {batch_idx} - optimizer_idx: {optimizer_idx}\n\n")
    #     optimizer.step(closure=optimizer_closure)

    # def lr_scheduler_step(
    #         self,
    #         scheduler: LRSchedulerTypeUnion,
    #         optimizer_idx: int,
    #         metric: Optional[Any],
    # ) -> None:
    #     print(f"\noptimizer_idx: {optimizer_idx} - ")

    def configure_optimizers(self):
        if self.hparams.tag_training:
            return self._configure_tgt_optimizers()
        else:
            return self._configure_std_optimizers()

    def _configure_tgt_optimizers(self):
        # optimizers
        desc_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.desc_encoder.parameters()), lr=self.hparams.desc_lr,
            betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        code_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.code_encoder.parameters()), lr=self.hparams.code_lr,
            betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        step_size_up = round(0.07 * self.trainer.estimated_stepping_batches)

        print(f"\nstep_size_up: {step_size_up}")
        print(f"estimated_stepping_batches: {self.trainer.estimated_stepping_batches}\n")

        desc_scheduler = torch.optim.lr_scheduler.CyclicLR(desc_optimizer, mode='triangular2',
                                                           base_lr=self.hparams.base_lr,
                                                           max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                                           cycle_momentum=False)
        code_scheduler = torch.optim.lr_scheduler.CyclicLR(code_optimizer, mode='triangular2',
                                                           base_lr=self.hparams.base_lr,
                                                           max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                                           cycle_momentum=False)
        # a tuple of:
        return (
            {"optimizer": desc_optimizer,
             "lr_scheduler": {"scheduler": desc_scheduler, "interval": "step", "name": "DESC_SCHDLR"},
             "frequency": 1},
            {"optimizer": code_optimizer,
             "lr_scheduler": {"scheduler": code_scheduler, "interval": "step", "name": "CODE_SCHDLR"},
             "frequency": 1},
        )

    def _configure_std_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        step_size_up = round(0.15 * self.trainer.estimated_stepping_batches)

        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2',
                                                      base_lr=self.hparams.base_lr,
                                                      max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                                      cycle_momentum=False)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "SCHDLR"}
        }
