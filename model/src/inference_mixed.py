import csv
import gc
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
import torch.distributed as dist
import transformers
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import LogitsProcessorList, LogitsProcessor

import wandb
from FinetuneDataset.fine_multi_dataset import FinetuneMultiDataset
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/gpfs/slayman/pi/gerstein/xt86/howard_dai/nlp-project/RadFM-finetune/src/Language_files")
    tokenizer_path: str = field(
        default='/gpfs/slayman/pi/gerstein/xt86/howard_dai/nlp-project/RadFM-finetune/src/Language_files',
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


WORLD_SIZE = torch.cuda.device_count()

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if len(sys.argv) < 2:
        print('Provide checkpoint #')

    ckpt_num = int(sys.argv[1])
    print(f'Checkpoint: {ckpt_num}')

    # Checkpoints loading part
    checkpoint_file = f'../output/finetuned-mixed-{ckpt_num}.pt'

    torch.multiprocessing.spawn(train,
                                args=(WORLD_SIZE, model_args.lang_encoder_path,
                                      model_args.tokenizer_path, checkpoint_file, ckpt_num),
                                nprocs=WORLD_SIZE,
                                join=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12344'
    # dist.init_process_group("nccl", rank=rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # print(f'Setup for rank {os.environ["LOCAL_RANK"]}')
    # dist.init_process_group(backend="nccl")


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, lang_encoder_path, tokenizer_path, checkpoint_file, ckpt_num):
    setup(rank, world_size)

    # Prepare model for DDP
    print(f'Loading LLaMA for {rank}')
    device = torch.device(f'cuda:{rank}')
    model = MultiLLaMAForCausalLM(lang_model_path=lang_encoder_path, device=device, peft=True, rank=rank)

    # Original
    # ckpt = torch.load('./Language_files/pytorch_model.bin', map_location='cpu')
    # model.load_state_dict(ckpt, strict=False)
    # model.make_peft()

    model = model.to(device)

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        consume_prefix_in_state_dict_if_present(checkpoint['model'], 'module.')
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f'Loaded checkpoint from {checkpoint_file}')
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    # model.unmerge()
    model.eval()

    if rank == 0:
        wandb.init(project="RadFM-inference", name=f'inference-mixed-{ckpt_num}', resume="never")

    print('Loading interpret-cxr-test-public dataset')
    dataset = datasets.load_dataset("StanfordAIMI/interpret-cxr-test-public")['test']
    val_dataset = FinetuneMultiDataset(text_tokenizer=tokenizer_path, dataset_path=None,
                                       split='test', dataset=dataset)

    # Prepare DataLoader with DistributedSampler
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            collate_fn=None,
                            num_workers=2,
                            sampler=val_sampler,
                            persistent_workers=True
                            )

    finding_id = val_dataset.text_tokenizer.encode("Finding:", add_special_tokens=False)
    impression_ids = val_dataset.text_tokenizer.encode("Impression", add_special_tokens=False)
    print(f'Finding ids: {finding_id}')
    print(f'Impression ids: {impression_ids}')
    # Custom Logits Processor to discourage generating specific sequence of tokens

    finding_id = torch.tensor(finding_id, device=device)

    class SequenceLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor):
            # print(f'Logits are {input_ids[0, -len(finding_id):]}')
            if torch.equal(input_ids[0, -len(finding_id):], finding_id):
                # print('Changing logits..')
                for idx in impression_ids:
                    logits[:, idx] = -float('Inf')
            return logits

    progress_bar = tqdm(val_loader, desc=f"Rank {rank} - Test", total=len(val_loader))
    file_name = f'../output/inference-mixed-{os.path.basename(checkpoint_file)}-rank{rank}.csv'
    with open(file_name, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["idx", "question", "answer", "prediction"])
        for i, item_raw in enumerate(progress_bar):
            item = {}
            for k, v in item_raw.items():
                if hasattr(v, 'to'):
                    item[k] = v.to(device)
                else:
                    item[k] = v

            question = item["question"]
            answer = item['answer']
            idx = item['idx'].item()
            lang_x = val_dataset.text_tokenizer(
                question, max_length=2048, truncation=True, return_tensors="pt"
            )['input_ids'].to(device)
            vision_x = item["vision_x"].to(device)

            with torch.no_grad():
                generation = model.generate(lang_x, vision_x)

            generated_text = val_dataset.text_tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

            writer.writerow([idx, question[0], answer[0], generated_text])
            if rank == 0:
                wandb.log({"idx": i})
            if i % 10 == 0:
                file.flush()

    cleanup()


if __name__ == "__main__":
    main()
