from transformers import (
    AutoConfig, 
    BartForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from pytorch_lightning import LightningModule 
from evaluate import load
import torch
import numpy as np

class ZeroShotClassificationModel(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ZeroShotClassificationModel")
        parser.add_argument("--batch_size", type=int, default=16)
        # parser.add_argument('--model_checkpoint', type = str, default = 'binhquoc/vie-deberta-small')
        return parent_parser

    def __init__(
        self,
        model_checkpoint: str,
        task_name: str = 'yahoo-topic',
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-6,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        seen_types: list = [],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        self.model = BartForSequenceClassification.from_pretrained(model_checkpoint, num_labels= 2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.metric_seen = load('glue', 'mrpc')
        self.metric_unseen = load('glue', 'mrpc')
    
    def forward(self, **inputs):
        return self.model(
            input_ids = inputs['input_ids'], 
            attention_mask =inputs['attention_mask'],
            labels = inputs['labels'],
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs[0:2]
        return {"loss": val_loss, 'batch': batch, 'logits': logits}

    def validation_epoch_end(self, validation_step_outputs):
        example_id = 0   
        for output_batch in validation_step_outputs:
            logits = output_batch['logits']
            batch = output_batch['batch']
            type_index = batch['type_index'].detach().cpu().numpy()
            predictions = torch.argmax(logits, dim=-1)
            references = batch['labels'].detach().cpu().numpy()
            seen_arr = []
            for type in type_index:
                if type in self.hparams.seen_types:
                    seen_arr.append(True)
                else:
                    seen_arr.append(False)
            unseen_arr = list(map(lambda x: not x,seen_arr))
            seen_predictions = predictions[seen_arr]
            unseen_predictions = predictions[unseen_arr]
            seen_references = references[seen_arr]
            unseen_references = references[unseen_arr]
            
            self.metric_seen.add_batch(predictions = seen_predictions, references = seen_references)
            self.metric_unseen.add_batch(predictions = unseen_predictions, references = unseen_references)

        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        seen_score = self.metric_seen.compute()
        unseen_score = self.metric_unseen.compute()
        self.log('seen_accuracy', seen_score['accuracy'], prog_bar=True)
        self.log('seen_f1', seen_score['f1'], prog_bar=True)
        self.log('unseen_accuracy', unseen_score['accuracy'], prog_bar=True)
        self.log('unseen_f1', unseen_score['f1'], prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight','LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.hparams.learning_rate, eps = self.hparams.adam_epsilon)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]