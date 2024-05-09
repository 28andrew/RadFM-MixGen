import torch

from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

# Load from pytorch_model.bin
model = MultiLLaMAForCausalLM(lang_model_path='/gpfs/slayman/pi/gerstein/xt86/howard_dai/nlp-project/RadFM-finetune/src/Language_files', device='cuda', peft=False)
ckpt = torch.load('./pytorch_model.bin', map_location='cpu')
model.load_state_dict(ckpt, strict=False)
# Save in HuggingFace model folder format
model.lang_model.save_pretrained('Base')