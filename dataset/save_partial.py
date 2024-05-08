from datasets import load_dataset, DatasetDict

print('Saving dataset without mimic')

dataset = load_dataset("StanfordAIMI/interpret-cxr-public", token='')

dataset_final = DatasetDict({"train": dataset["train"],
                             "validation": dataset["validation"]})

dataset_final.save_to_disk("./partial_data")