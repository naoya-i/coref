from bert import tokenization
import json

genre = "nw"
model_name = "spanbert_base"
text = """
Edmund Spenser (/ˈspencer/; 1552/1553 – 13 January 1599) was an English poet best known for The Faerie Queene, an epic poem and fantastical allegory celebrating the Tudor dynasty and Elizabeth I.
He is recognized as one of the premier craftsmen of nascent Modern English verse, and is often considered one of the greatest poets in the English language.
""".strip().split("\n")

data = {
    'doc_key': genre,
    'sentences': [["[CLS]"]],
    'speakers': [["[SPL]"]],
    'clusters': [],
    'sentence_map': [0],
    'subtoken_map': [0],
}

# Determine Max Segment
max_segment = None
for line in open('experiments.conf'):
    if line.startswith(model_name):
        max_segment = True
    elif line.strip().startswith("max_segment_len"):
        if max_segment:
            max_segment = int(line.strip().split()[-1])
            break

tokenizer = tokenization.FullTokenizer(vocab_file="cased_config_vocab/vocab.txt", do_lower_case=False)
subtoken_num = 0
for sent_num, line in enumerate(text):
    raw_tokens = line.split()
    tokens = tokenizer.tokenize(line)
    if len(tokens) + len(data['sentences'][-1]) >= max_segment:
        data['sentences'][-1].append("[SEP]")
        data['sentences'].append(["[CLS]"])
        data['speakers'][-1].append("[SPL]")
        data['speakers'].append(["[SPL]"])
        data['sentence_map'].append(sent_num - 1)
        data['subtoken_map'].append(subtoken_num - 1)
        data['sentence_map'].append(sent_num)
        data['subtoken_map'].append(subtoken_num)

    ctoken = raw_tokens[0]
    cpos = 0
    for token in tokens:
        data['sentences'][-1].append(token)
        data['speakers'][-1].append("-")
        data['sentence_map'].append(sent_num)
        data['subtoken_map'].append(subtoken_num)
        
        if token.startswith("##"):
            token = token[2:]
        if len(ctoken) == len(token):
            subtoken_num += 1
            cpos += 1
            if cpos < len(raw_tokens):
                ctoken = raw_tokens[cpos]
        else:
            ctoken = ctoken[len(token):]

data['sentences'][-1].append("[SEP]")
data['speakers'][-1].append("[SPL]")
data['sentence_map'].append(sent_num - 1)
data['subtoken_map'].append(subtoken_num - 1)

with open("/tmp/sample.in.json", 'w') as out:
    json.dump(data, out, sort_keys=True)

