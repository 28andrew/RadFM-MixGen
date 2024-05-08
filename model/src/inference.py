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


# @dataclass
# class DataCollator(object):
#
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         # print(instances) 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words
#         vision_xs, lang_xs, attention_masks, labels, loss_reweight, questions, answers, idxs = tuple(
#             [instance[key] for instance in instances] for key in
#             ('vision_x', 'lang_x', 'attention_mask', 'labels', 'loss_reweight', 'question', 'answer', 'idx'))
#
#         lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
#         attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks], dim=0)
#         labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
#         loss_reweight = torch.cat([_.unsqueeze(0) for _ in loss_reweight], dim=0)
#         # print(lang_xs.shape,attention_masks.shape,labels.shape)
#
#         target_H = 512
#         target_W = 512
#         target_D = 4
#         MAX_D = 0
#
#         D_list = list(range(4, 65, 4))
#         if len(vision_xs) == 1:
#             if vision_xs[0].shape[0] > 6:
#                 D_list = list(range(4, 33, 4))
#
#         for ii in vision_xs:
#             try:
#                 D = ii.shape[-1]
#                 if D > MAX_D:
#                     MAX_D = D
#             except:
#                 continue
#         for temp_D in D_list:
#             if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
#                 target_D = temp_D
#
#         if len(vision_xs) == 1 and target_D > 4:
#             target_H = 256
#             target_W = 256
#
#         vision_xs = [torch.nn.functional.interpolate(s, size=(target_H, target_W, target_D)) for s in vision_xs]
#
#         vision_xs = torch.nn.utils.rnn.pad_sequence(
#             vision_xs, batch_first=True, padding_value=0
#         )
#         # print(vision_xs.shape, vision_xs.dtype)
#         return dict(
#             lang_x=lang_xs,
#             vision_x=vision_xs,
#             attention_mask=attention_masks,
#             labels=labels,
#             loss_reweight=loss_reweight,
#             question=questions,
#             answer=answers,
#             idx=idxs
#         )


def compute_metrics(eval_preds):
    # metric = load_metric("glue", "mrpc")
    ACCs = eval_preds.predictions
    # print(ACCs)
    return {"accuracy": np.mean(ACCs, axis=-1)}

WORLD_SIZE = torch.cuda.device_count()


def main():
    # for arg in sys.argv:
    #     if arg.startswith("--local-rank="):
    #         rank = arg.split("=")[1]
    #         sys.argv.remove(arg)
    #         sys.argv.append('--local_rank')
    #         sys.argv.append(rank)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if len(sys.argv) < 2:
        print('Provide checkpoint #')

    ckpt_num = int(sys.argv[1])
    print(f'Checkpoint: {ckpt_num}')

    # Checkpoints loading part
    checkpoint_file = f'../output/finetuned-{ckpt_num}.pt'

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
    model = MultiLLaMAForCausalLM(lang_model_path=lang_encoder_path, device=device, peft=False, rank=rank)

    # Original
    ckpt = torch.load('./Language_files/pytorch_model.bin', map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    del ckpt
    gc.collect()
    # model.make_peft()

    model = model.to(device)

    # if checkpoint_file:
    #     checkpoint = torch.load(checkpoint_file, map_location='cpu')
    #     consume_prefix_in_state_dict_if_present(checkpoint['model'], 'module.')
    #     model.load_state_dict(checkpoint['model'], strict=True)
    #     print(f'Loaded checkpoint from {checkpoint_file}')
    #     del checkpoint
    #     gc.collect()
    #     torch.cuda.empty_cache()

    # model.unmerge()
    model.eval()

    if rank == 0:
        wandb.init(project="RadFM-inference", name=f'inference-{ckpt_num}', resume="never")

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

    # Tokenize the phrases and convert to input IDs
    bad_words_ids = []
    # unwanted = ['nan', 'Finding: Impression: ']
    # for phrase in unwanted:
    #     ids = val_dataset.text_tokenizer(phrase, add_special_tokens=False).input_ids
    #     bad_words_ids.append(ids)

    # Token
    # ID: 383 - Token: | F |
    # Token
    # ID: 4015 - Token: | inding |
    # Token
    # ID: 29901 - Token: |: |
    # Token
    # ID: 1954 - Token: | Im |
    # Token
    # ID: 2590 - Token: | pression |
    # Token
    # ID: 29901 - Token: |: |

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
    file_name = f'../output/inference-{os.path.basename(checkpoint_file)}-rank{rank}.csv'
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
                generation = model.generate(lang_x, vision_x,
                                            # logits_processor=LogitsProcessorList([SequenceLogitsProcessor()])
                                            )

            # print('\n')
            # for token_id in generation[0]:
            #     token = val_dataset.text_tokenizer.decode([token_id])
            #     print(f'Token ID: {token_id} - Token: |{token}|')
            # print('\n')

            generated_text = val_dataset.text_tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
            print(generated_text)

            writer.writerow([idx, question[0], answer[0], generated_text])
            if rank == 0:
                wandb.log({"idx": i})
            if i % 10 == 0:
                file.flush()

    cleanup()


if __name__ == "__main__":
    main()
