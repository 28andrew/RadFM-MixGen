from torch import nn
from transformers import PreTrainedModel, BitsAndBytesConfig
from transformers.models.llama import LlamaForCausalLM
from .my_embedding_layer import MyEmbedding
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import tqdm.auto as tqdm
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import numpy as np


from finetune_peft import get_peft_config, CastOutputToFloat, save_tunable_parameters
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType, prepare_model_for_kbit_training,
)

from accelerate import init_empty_weights

class MultiLLaMAForCausalLM(nn.Module):
    def __init__(self, lang_model_path, device=None, peft=True, rank=0):
        super(MultiLLaMAForCausalLM, self).__init__()

        # quant_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        # Load on rank's GPU..
        num_gpus = torch.cuda.device_count()
        mem_map = {}
        for i in range(num_gpus):
            mem_map[i] = "0GB"
        mem_map[rank] = '80GB'
        # with init_empty_weights():

        self.peft = peft
        if self.peft:
            self.lang_model = LlamaForCausalLM.from_pretrained(
                lang_model_path,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map='sequential',
                max_memory=mem_map,
                # offload_folder=f"offload-{rank}",
                # offload_state_dict=True,
                # torch_dtype=torch.bfloat16,
            )
            self.make_peft()
        else:
            self.lang_model = LlamaForCausalLM.from_pretrained(
                lang_model_path,
                device_map='sequential',
                max_memory=mem_map
            )

        self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()

        # self.lang_model.requires_grad_(False)
        self.embedding_layer = MyEmbedding()
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        self.hidden_dim = 5120
        self.voc_size = 32000

    def make_peft(self):
        self.lang_model = prepare_model_for_kbit_training(self.lang_model)

        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.1,
            target_modules=['gate_proj', 'down_proj', 'up_proj', 'q_proj', 'v_proj', 'k_proj', 'o_proj'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.lang_model = get_peft_model(self.lang_model, lora_config)
        # self.lang_model.config.use_cache = False
        self.lang_model.print_trainable_parameters()

    def unmerge(self):
        self.lang_model = self.lang_model.merge_and_unload()

    def forward(self,lang_x, vision_x, attention_mask, labels, loss_reweight,key_words_query=None):
        if labels.shape == lang_x.shape:
            self.embedding_layer.flag = 'Text'
            # lang_x = lang_x.to(vision_x.dtype)
            # lang_x = lang_x + torch.zeros(1, dtype=lang_x.dtype, device=lang_x.device, requires_grad=True)
            # vision_x = vision_x + torch.zeros(1, dtype=vision_x.dtype, device=vision_x.device, requires_grad=True) 
            # input_embedding = checkpoint(self.embedding_layer, lang_x, vision_x)
            input_embedding,loss_match= self.embedding_layer(lang_x, vision_x,key_words_query)   # ,loss_matching
            # print(f'input embedding: {input_embedding}')
            # print(f'attention_mask: {attention_mask}')
            # print(f'labels: {labels}')
            output = self.lang_model(inputs_embeds = input_embedding,attention_mask = attention_mask, labels = labels)
            # print(f'llama output: {output}')
            logits = output['logits']

            loss_reg = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_loss_reweight = loss_reweight[...,1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction = 'none')
                shift_logits = shift_logits.view(-1, self.voc_size)
                shift_labels = shift_labels.view(-1)
                shift_loss_reweight = shift_loss_reweight.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                shift_loss_reweight = shift_loss_reweight.to(shift_logits.device) 
                loss_reg = loss_fct(shift_logits, shift_labels)
                loss_reg = torch.sum(shift_loss_reweight*loss_reg)/torch.sum(shift_loss_reweight)
            loss = loss_reg
            if loss_match!= None:
                loss = 0.8*loss + 0.2*loss_match
            
            logits = output['logits'][..., :-1, :].contiguous().detach()
            total = len(labels)
            predictions = torch.argmax(logits, dim=-1)
            labels = labels[..., 1:].contiguous()
            Acc = torch.sum(torch.all(torch.logical_or(predictions == labels, labels == -100),dim = -1))       
            Accuracy = Acc /total
            # print(f'llama output["loss"]: {output["loss"]}')
            return dict(
                # loss_reg = loss_reg,
                # loss_matching = loss_matching,
                logits =  Accuracy,
                loss = output['loss'],
            )
        ### useless for now ignore the folowing codes ###
        # if labels.shape == vision_x.shape:
        #    self.embedding_layer.flag = 'Seg'
        #    input_embedding = self.embedding_layer(lang_x, vision_x)
    
    def generate(self, lang_x,vision_x, **kwargs):
        self.embedding_layer.flag = 'Text'
        with torch.no_grad():
            input_embedding,_ = self.embedding_layer(lang_x, vision_x) 
            generation = self.lang_model.generate(inputs_embeds = input_embedding, max_new_tokens =200,top_k=50, **kwargs)
        return generation
