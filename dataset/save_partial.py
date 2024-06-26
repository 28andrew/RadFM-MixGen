from datasets import load_dataset, DatasetDict

print('Saving dataset without mimic')

# Load the dataset
dataset = load_dataset("StanfordAIMI/interpret-cxr-public", token='')
# Make into HuggingFace format
dataset_final = DatasetDict({"train": dataset["train"],
                             "validation": dataset["validation"]})
# Save to disk
dataset_final.save_to_disk("./partial_data")