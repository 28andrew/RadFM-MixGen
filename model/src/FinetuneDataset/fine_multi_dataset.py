import gc
import random

import datasets
import spacy
import torch
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
from torchvision import transforms
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

transform = transforms.Compose([
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])


def get_image(pil_img):
    image = pil_img.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(-1)
    return image


captions = [
            # "Can you provide a caption consists of finding and impression for this medical image?",
            "Describe the finding and impression of the medical image you see.",
            # "Please caption this medical scan with finding and impression.",
            # "What is the finding and impression of this image?",
            # "Describe this medical scan with finding and impression.",
            # "Please write a caption consists of finding and impression for this image.",
            # "Can you summarize with finding and impression the images presented?",
            # "Please caption this scan with finding and impression.",
            # "Please provide a caption consists of finding and impression for this medical image.",
            # "Can you provide a summary consists of finding and impression of this radiograph?",
            # "What are the findings and impression presented in this medical scan?",
            # "Please write a caption consists of finding and impression for this scan.",
            # "Can you provide a description consists of finding and impression of this medical scan?",
            # "Please caption this medical scan with finding and impression.",
            # "Can you provide a caption consists of finding and impression for this medical scan?"
]

captions_finding = [
            # "Can you provide a caption consists of finding for this medical image?",
            "Describe the finding of the medical image you see.",
            # "Please caption this medical scan with finding.",
            # "What is the finding of this image?",
            # "Describe this medical scan with finding.",
            # "Please write a caption consists of finding for this image.",
            # "Can you summarize with finding the images presented?",
            # "Please caption this scan with finding.",
            # "Please provide a caption consists of finding for this medical image.",
            # "Can you provide a summary consists of finding of this radiograph?",
            # "What are the findings presented in this medical scan?",
            # "Please write a caption consists of finding for this scan.",
            # "Can you provide a description consists of finding of this medical scan?",
            # "Please caption this medical scan with finding.",
            # "Can you provide a caption consists of finding for this medical scan?"
]

captions_impression = [
            # "Can you provide a caption consists of impression for this medical image?",
            "Describe the impression of the medical image you see.",
            # "Please caption this medical scan with impression.",
            # "What is the impression of this image?",
            # "Describe this medical scan with impression.",
            # "Please write a caption consists of impression for this image.",
            # "Can you summarize with impression the images presented?",
            # "Please caption this scan with impression.",
            # "Please provide a caption consists of impression for this medical image.",
            # "Can you provide a summary consists of impression of this radiograph?",
            # "What are the impressions presented in this medical scan?",
            # "Please write a caption consists of impression for this scan.",
            # "Can you provide a description consists of impression of this medical scan?",
            # "Please caption this medical scan with impression.",
            # "Can you provide a caption consists of impression for this medical scan?"
]

import pickle


class MixgenDataset(Dataset):
    def __init__(self, dataset_path, split, mixgen_pickle_paths, mixgen_ratio=0.2):
        self.mixgen_data = []
        for path in mixgen_pickle_paths:
            with open(path, 'rb') as file:
                data = pickle.load(file)
                self.mixgen_data.extend(data)
            gc.collect()
        self.mixgen_length = len(self.mixgen_data)
        self.normal_data = datasets.load_from_disk(dataset_path)[split]
        self.normal_length = len(self.normal_data)
        self.mixgen_ratio = mixgen_ratio
        gc.collect()

    def __len__(self):
        return int(1e9)

    def __getitem__(self, item):
        if random.random() <= self.mixgen_ratio:
            return self.mixgen_data[random.randint(0, self.mixgen_length - 1)]
        else:
            return self.normal_data[random.randint(0, self.normal_length - 1)]


class InterpretCXRDataset(Dataset):
    def __init__(self, dataset_path, split, dataset=None):
        if dataset:
            self.dataset = dataset
        else:
            self.dataset = datasets.load_from_disk(dataset_path)[split]
        self.data = self.dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # image is a tensor of shape [c,w,h,d], c is channel=3, w is width, h is height, d is depth(1 for chestxray,pmcoa,pmcvqa)

        # image = (image - image.min()) / (image.max() - image.min())
        # contain_nan = (True in np.isnan(image))
        # if contain_nan:
        #     image = np.random.randn(3, 512, 512, 4)

        # image = torch.from_numpy(image).float()
        finding = item['findings']
        impression = item['impression']
        if finding and impression:
            question = random.choice(captions)
            answer = 'Finding: ' + str(finding) + 'Impression: ' + str(impression)
        elif finding:
            question = random.choice(captions_finding)
            answer = 'Finding: ' + str(finding)
        else:
            question = random.choice(captions_impression)
            answer = 'Impression: ' + str(impression)

        image_dict = []

        # position = 0 if (random.random() < 0.5) else len(question)
        position = 0
        if not hasattr(item['images'], '__iter__'):
            item['images'] = [item['images']]
        for image in item['images']:
            dict_idx = {
                "image": get_image(image),
                "position": {
                    "question": position
                }
            }
            image_dict.append(dict_idx)

        return {
            "image_dict": image_dict,
            "question": question,
            "answer": answer,
        }


WORDS_EXTRACT = None

class umls_extractor:
    def __init__(self):
        nlp = spacy.load("en_core_sci_lg")
        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.nlp = nlp

    def extract(self, text):
        doc = self.nlp(text)
        ent_set = doc.ents
        return ent_set


def get_word_extractor():
    global WORDS_EXTRACT
    if not WORDS_EXTRACT:
        WORDS_EXTRACT = umls_extractor()
    return WORDS_EXTRACT


def find_position(label, key_embeddings):
    loss_reweight = torch.ones(label.shape)
    for i in range(len(label)):
        if label[i] == -100:
            loss_reweight[i] = 0
        else:
            for key_embedding in key_embeddings:
                if torch.equal(label[i:i+len(key_embedding)], key_embedding):
                    loss_reweight[i:i+len(key_embedding)] = 3
    return loss_reweight


def stack_images(images):
    target_H = 512
    target_W = 512
    target_D = 4
    if len(images) == 0:
        return torch.zeros((1, 3, target_H, target_W, target_D))
    MAX_D = 4
    D_list = list(range(4, 65, 4))

    for ii in images:
        try:
            D = ii.shape[3]
            if D > MAX_D:
                MAX_D = D
        except:
            continue
    for temp_D in D_list:
        if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
            target_D = temp_D

    stack_images = []
    for s in images:
        if len(s.shape) == 3:
            # print(s.shape)
            stack_images.append(
                torch.nn.functional.interpolate(s.unsqueeze(0).unsqueeze(-1), size=(target_H, target_W, target_D)))
        else:
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0), size=(target_H, target_W, target_D)))
    images = torch.cat(stack_images, dim=0)
    return images


class FinetuneMultiDataset(Dataset):
    def __init__(self, text_tokenizer, dataset_path, split, max_seq = 2048, max_img_size = 100,
                 image_num=32,voc_size =32000, eval=False, dataset=None):
        self.text_tokenizer = text_tokenizer
        self.max_img_size = max_img_size
        self.image_num = image_num
        self.max_seq = max_seq
        self.voc_size = voc_size
        self.H = 512
        self.W = 512
        self.image_padding_tokens = []
        self.eval = eval
        if isinstance(self.text_tokenizer,str):
            self.text_tokenizer = LlamaTokenizer.from_pretrained(
                self.text_tokenizer,
            )
            special_token = {"additional_special_tokens": ["<image>","</image>"]}
            for i in range(max_img_size):
                image_padding_token = ""
                for j in range(image_num):
                    image_token = "<image"+str(i*image_num+j)+">"
                    image_padding_token = image_padding_token + image_token
                    special_token["additional_special_tokens"].append("<image"+str(i*image_num+j)+">")
                self.image_padding_tokens.append(image_padding_token)
            self.text_tokenizer.add_special_tokens(
                special_token
            )
            self.text_tokenizer.pad_token_id = 0
            self.text_tokenizer.bos_token_id = 1
            self.text_tokenizer.eos_token_id = 2

        self.data_whole_2D = []
        self.data_whole_3D = []
        self.dataset_reflect = {}

        interpret_cxr_dataset = InterpretCXRDataset(dataset_path, split, dataset=dataset)
        self.dataset_reflect['interpret_cxr_dataset'] = interpret_cxr_dataset
        self.data_whole_3D = self.data_whole_3D + [{'interpret_cxr_dataset': i} for i in
                                                   range(len(interpret_cxr_dataset))]

        self.data_whole = self.data_whole_2D + self.data_whole_3D

    def __len__(self):
        return len(self.data_whole)

    def __getitem__(self, idx):
        # vision_x, lang_x, attention_mask, labels
        sample = list(self.data_whole[idx].items())[0]
        # print(sample)
        dataset_index = sample[0]
        sample = self.dataset_reflect[sample[0]][sample[1]]
        '''
                Dict: {
                    "image_dict": [
                                    {"image": image, # image is a tensor of shape [c,w,h,d], c is channel=3, w is width, h is height, d is depth(1 for chestxray,pmcoa,pmcvqa)
                                    "position": {"question": 0}}, position is a dict, random choice of 0 or len(question)
                                ]
                    "question": question,
                    "answer":answer,
                    }
                '''
        images = sample["image_dict"]
        question = sample["question"]
        answer = sample["answer"]
        # if idx == 0:
        #     print(f']Sample {sample}')
        images, question, answer = self.text_add_image(images, question, answer)

        # print(question,answer)
        # print(f'Images shape: {sample["image_dict"][0]["image"].shape}')
        ### make vision_x
        vision_x = stack_images(images)

        # if idx == 0:
        #     print(f']Vision x shape: {vision_x.shape}')

        ### make lang_x ###
        self.text_tokenizer.padding_side = "right"
        if not self.eval:
            text_tensor = self.text_tokenizer(
                question + ' ' + answer, max_length=self.max_seq, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            lang_x = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]
            try:
                lang_x[torch.sum(attention_mask)] = self.text_tokenizer.eos_token_id
            except:
                pass

            emphasize_words = []
            emphasize_words = [str(_) for _ in get_word_extractor().extract(answer)]

            if emphasize_words != []:
                emphasize_words_tensor = self.text_tokenizer(
                    emphasize_words, max_length=self.max_seq
                )
                key_embeddings = [torch.tensor(_[1:]) for _ in emphasize_words_tensor['input_ids']]
            else:
                key_embeddings = []
            question_tensor = self.text_tokenizer(
                question, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
            )
            question_length = torch.sum(question_tensor["attention_mask"][0])
            labels = lang_x.clone()
            labels[labels == self.text_tokenizer.pad_token_id] = -100
            labels[labels >= self.voc_size] = -100
            labels[:question_length] = -100

            reweight_tensor = find_position(labels, key_embeddings)
            if dataset_index == 'paper_inline_dataset':
                emphasize_words = []
            return {'vision_x': vision_x, 'lang_x': lang_x, 'attention_mask': attention_mask, 'labels': labels,
                    'loss_reweight': reweight_tensor,
                    # 'key_words_query': emphasize_words,
                    'question': question, 'answer': answer, 'idx': idx}
        else:
            return {'vision_x': vision_x, 'question': question, 'answer': answer, 'idx': idx}
        ### make label ###
        # print(labels,key_embeddings,reweight_tensor)
        # print(question)

    def text_add_image(self,images,question,answer):
        ref_image = []
        question = str(question)
        answer = str(answer)
        question_list = [[] for _ in range(len(str(question)))]
        answer_list = [[] for _ in range(len(str(answer)))]
        for index, image in enumerate(images):
            ref_image.append(image["image"])
            position = image["position"]
            position = list(position.items())[0]
            if position[0] == 'question':
                insert_loc = position[1] -1
                if insert_loc < 0:
                    insert_loc = 0
                question_list[insert_loc].append(index)
            if position[0] == 'answer':
                insert_loc = position[1] -1
                if insert_loc < 0:
                    insert_loc = 0
                answer_list[insert_loc].append(index)
        new_question = ''
        new_answer = ''
        question = str(question)
        for char_i in range(len(question)):
            if question_list[char_i] == []:
                new_question = new_question + question[char_i]
            if question_list[char_i] != []:
                for img_index in question_list[char_i]:
                    try:
                        new_question = new_question + '<image>' + self.image_padding_tokens[img_index] + '</image>'
                    except:
                        print("Error: out of max image input size")
                new_question = new_question + question[char_i]
        answer = str(answer)
        for char_i in range(len(str(answer))):
            if answer_list[char_i] == []:
                new_answer = new_answer + answer[char_i]
            if answer_list[char_i] != []:
                for img_index in answer_list[char_i]:
                    try:
                        new_answer = new_answer + '<image>' + self.image_padding_tokens[img_index] + '</image>'
                    except:
                        print("Error: out of max image input size")
                new_answer = new_answer + answer[char_i]
        new_answer = new_answer.replace('•','')
        return ref_image,new_question,new_answer
