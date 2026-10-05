from transformers import BertTokenizer
import json
import re

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

with open('ner_sample.json', 'r', encoding='utf-8') as file:
    sample_json = json.load(file)

texts_array_flat = sample_json["texts"]

texts = []

for sentence in texts_array_flat:
    texts.append(re.split(r'[,\:\s]+', sentence))

label2id = {"email": 0, "website": 1, "phone_number": 2, "fax": 3, "other": 4}

labels = []

counter = 0
for sentence in texts:
    labels.append([])
    for word in sentence:
        if word in sample_json["openai_emails"]:
            labels[counter].append(label2id["email"])
            print("email", word)
        elif word in sample_json["openai_websites"]:
            labels[counter].append(label2id["website"])
            print("website", word)
        elif word in [item["Number"] for item in sample_json["openai_phone_numbers"]]:
            labels[counter].append(label2id["phone_number"])
            print("phone", word)
        elif word in [item["Number"] for item in sample_json["openai_fax"]]:
            labels[counter].append(label2id["fax"])
            print("fax", word)
        else:
            labels[counter].append(label2id["other"])
    counter = counter + 1