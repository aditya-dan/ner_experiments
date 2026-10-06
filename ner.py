from transformers import BertTokenizer
import json
import re
import itertools

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
            labels[counter].append("email")
            print("email", word)
        elif word in sample_json["openai_websites"]:
            labels[counter].append("website")
            print("website", word)
        elif word in [item["Number"] for item in sample_json["openai_phone_numbers"]]:
            labels[counter].append("phone_number")
            print("phone", word)
        elif word in [item["Number"] for item in sample_json["openai_fax"]]:
            labels[counter].append("fax")
            print("fax", word)
        else:
            labels[counter].append("other")
    counter = counter + 1

words = list(itertools.chain.from_iterable(texts))
word_labels = list(itertools.chain.from_iterable(labels))

encoding = tokenizer(
    words,
    is_split_into_words=True,
    truncation=True,
    padding="max_length",
    max_length=128
)

word_ids = encoding.word_ids()

token_labels = []

for word_id in word_ids:
    if word_id is None:
        token_labels.append(-100)
    else:
        label = word_labels[word_id]
        token_labels.append(label2id[label])
