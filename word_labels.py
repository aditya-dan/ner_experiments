from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

label2id = {"name": 0, "website": 1, "phone_number": 2, "fax_number": 3}

words_labels_dict = {"Aditya Dan": 0, "www.jkspdcl.nic.in": 1, "0194-2458005": 2, "0191-2430548": 2, "0194-2451665": 3, "0191-2435403": 3}

words = list(words_labels_dict.keys())

word_labels = [words_labels_dict[word] for word in words]

for word in words:
    print(tokenizer.tokenize(word))

encoding = tokenizer(
    words,
    is_split_into_words=True
)

word_ids = encoding.word_ids()

token_labels = []

for word_id in word_ids:
    if word_id is None:
        token_labels.append(-100)
    else:
        token_labels.append(word_labels[word_id])

print(token_labels)

