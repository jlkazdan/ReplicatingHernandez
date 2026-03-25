from transformers import Trainer
import torch, torch.nn.functional as F


# Copied from https://discuss.huggingface.co/t/working-with-olmo2-want-to-know-if-z-loss-is-implemented/168522/2
# TODO: Fix this. It currently throws an error:
# [rank0]: TypeError: ZLossTrainer.compute_loss() got an unexpected keyword argument 'num_items_in_batch'
class ZLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # [B,T,V]

        # standard CE on shifted tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        # z-loss (penalize logsumexp of logits at label positions)
        with torch.no_grad():
            valid = shift_labels.ne(-100)
        z = torch.logsumexp(shift_logits, dim=-1)  # [B,T-1]
        z = z.masked_select(valid)
        z_loss = (
            (z**2).mean() if z.numel() else torch.tensor(0.0, device=logits.device)
        )

        lam = 1e-4  # typical coefficient from T5/PaLM-style recipes
        loss = ce + lam * z_loss
        return (loss, outputs) if return_outputs else loss
