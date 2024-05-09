import gc
import glob
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict

import numpy as np
import torch
import torch.distributed as dist
import transformers
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from FinetuneDataset.fine_multi_dataset import FinetuneMultiDataset, MixgenDataset
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

# Dataclasses for model, data, and training arguments
@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="./Base/")
    tokenizer_path: str = field(
        default='./Language_files',
        metadata={"help": "Path to the tokenizer data."})


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=1)
    batch_size_3D: int = field(default=1)
    output_dir: Optional[str] = field(
        default="/gpfs/slayman/pi/gerstein/xt86/howard_dai/nlp-project/RadFM-finetune/output")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


# Collate together data fro training
@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(instances) 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words
        vision_xs, lang_xs, attention_masks, labels, loss_reweight = tuple(
            [instance[key] for instance in instances] for key in
            ('vision_x', 'lang_x', 'attention_mask', 'labels', 'loss_reweight'))

        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        loss_reweight = torch.cat([_.unsqueeze(0) for _ in loss_reweight], dim=0)
        # print(lang_xs.shape,attention_masks.shape,labels.shape)

        # Standardize image stack sizes
        target_H = 512
        target_W = 512
        target_D = 4
        MAX_D = 0

        D_list = list(range(4, 65, 4))
        if len(vision_xs) == 1:
            if vision_xs[0].shape[0] > 6:
                D_list = list(range(4, 33, 4))

        for ii in vision_xs:
            try:
                D = ii.shape[-1]
                if D > MAX_D:
                    MAX_D = D
            except:
                continue
        for temp_D in D_list:
            if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
                target_D = temp_D

        if len(vision_xs) == 1 and target_D > 4:
            target_H = 256
            target_W = 256

        vision_xs = [torch.nn.functional.interpolate(s, size=(target_H, target_W, target_D)) for s in vision_xs]

        # Pad vision inputs
        vision_xs = torch.nn.utils.rnn.pad_sequence(
            vision_xs, batch_first=True, padding_value=0
        )
        # print(vision_xs.shape, vision_xs.dtype)
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            attention_mask=attention_masks,
            labels=labels,
            loss_reweight=loss_reweight
        )

# Hyperparameters
NUM_EPOCHS = 1
LEARNING_RATE = 0.0002
# Save setting
SAVE_STEPS = 250

# For DDP
WORLD_SIZE = torch.cuda.device_count()


def main():
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Checkpoints loading part
    checkpoint_files = glob.glob('../output/finetuned-mixed-*.pt')
    checkpoint_file = None
    if checkpoint_files:
        checkpoint_file = max(checkpoint_files,
                                      key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))
        print(f"Resuming from checkpoint {checkpoint_file}")

    # train(os.environ['LOCAL_RANK'], model_args.lang_encoder_path,
    #                                   model_args.tokenizer_path, global_step_start, start_epoch)
    torch.multiprocessing.spawn(train,
                                args=(WORLD_SIZE, model_args.lang_encoder_path,
                                      model_args.tokenizer_path, checkpoint_file),
                                nprocs=WORLD_SIZE,
                                join=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # dist.init_process_group("nccl", rank=rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # print(f'Setup for rank {os.environ["LOCAL_RANK"]}')
    # dist.init_process_group(backend="nccl")


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, lang_encoder_path, tokenizer_path, checkpoint_file):
    setup(rank, world_size)

    # Load dataset settings
    print('Loading dataset')
    dataset_path = '/gpfs/slayman/pi/gerstein/xt86/howard_dai/nlp-project/dataset/partial_data'

    if rank == 0:
        pickle_paths = ['../../dataset/mixgen1.pkl'
            # , '../../dataset/mixgen2.pkl'
                        ]
    else:
        pickle_paths = ['../../dataset/mixgen2.pkl'
            # , '../../dataset/mixgen4.pkl'
                        ]

    # Prepare model for DDP
    print(f'Loading LLaMA for {rank}')
    device = torch.device(f'cuda:{rank}')

    start_epoch = 0
    global_step = 0
    if checkpoint_file:
        # If the model is resumed from a checkpoint, load it
        model = MultiLLaMAForCausalLM(lang_model_path=lang_encoder_path, device=device, peft=True, rank=rank)
        model = model.to(device)
        model = DDP(model, device_ids=[rank])

        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        run_id = checkpoint['run_id']
    else:
        # Load from scratch
        print('Starting model from scratch..')
        model = MultiLLaMAForCausalLM(lang_model_path='./Base/', device=device, peft=True, rank=rank)
        # ckpt = torch.load('./Language_files/pytorch_model.bin', map_location='cpu')
        # model.load_state_dict(ckpt, strict=False)
        # model.make_peft()
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
        run_id = wandb.util.generate_id()

    if rank == 0:
        # Use wandb in the first process
        wandb.init(project="RadFM-finetune", id='mixed-' + run_id, resume='allow')

    if checkpoint_file:
        # Load actual checkpoint
        model.load_state_dict(checkpoint['model'], strict=False)
        del checkpoint['model']
        gc.collect()

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # optimizer = bnb.optim.PagedAdam8bit(model.parameters(), lr=LEARNING_RATE)

    # Load dataset
    mix_dataset = MixgenDataset(dataset_path, 'train', pickle_paths, 0.15)
    train_dataset = FinetuneMultiDataset(text_tokenizer=tokenizer_path, dataset_path=dataset_path, split='train',
                                         dataset=mix_dataset)

    # Prepare DataLoader. No DistributedSampler needed for MixGen data
    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=DataCollator(),
                              num_workers=0,
                              # sampler=train_sampler,
                              # persistent_workers=True
                              )

    if checkpoint_file:
        # Restore optimizer state
        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    batches_per_epoch = len(train_loader)
    model.train()
    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        if global_step > epoch * batches_per_epoch:
            batches_to_skip = global_step - epoch * batches_per_epoch
        else:
            batches_to_skip = 0

        total_batches_to_process = batches_per_epoch - batches_to_skip

        progress_bar = tqdm(train_loader, desc=f"Rank {rank} - Epoch {epoch}", total=total_batches_to_process)
        for i, batch in enumerate(progress_bar, start=1):
            try:
                # Skip batches on resume
                if i > total_batches_to_process:
                    break

                # Convert tensors to device
                batch_converted = {}
                for k, v in batch.items():
                    if hasattr(v, 'to'):
                        batch_converted[k] = v.to(device)
                    else:
                        batch_converted[k] = v

                # Forward
                outputs = model(**batch_converted)
                loss = outputs['loss']

                # Reverse
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()

                # torch.cuda.empty_cache()

                if rank == 0:
                    # WANDB logging on first process
                    wandb.log({"global_step": global_step, "epoch": epoch, "loss": loss.item()})

                    # Save every SAVE_STEPS steps
                    if global_step > 0 and global_step % SAVE_STEPS == 0:
                        print(f"Saving checkpoint at step {global_step}")
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            # 'scheduler': lr_scheduler.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch,
                            'run_id': run_id
                        }
                        torch.save(checkpoint, f'../output/finetuned-mixed-{global_step}.pt')

                global_step += 1
            except Exception as e:
                # Handle exceptions such as OOM gracefully..
                print(f'[{rank}] Step failed: {e}')
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                pass

    cleanup()


if __name__ == "__main__":
    main()
