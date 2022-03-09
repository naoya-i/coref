# spanbert-coref

Coreference resolution system powered by BERT/SpanBERT. This is a thin wrapper for https://github.com/mandarjoshi90/coref.


## Setup

1. Make sure TensorFlow works

Please make sure that `tensorflow >= 1.14` or `tensorflow == 2.x` (preferably GPU version) works on your machine.
If you manually install TensorFlow, we recommend creating a dedicated virtual environment and following https://www.tensorflow.org/install.

See https://www.tensorflow.org/guide/gpu#setup for testing GPU support by TensorFlow.


2. Install the software

```console
foo@bar:~$ pip install git+https://github.com/naoya-i/spanbert-coref
```


3. Run setup scripts

```console
foo@bar:~$ python -m spanbertcoref.configure
```

It compiles a TensorFlow kernel and copies a basic configuration file to your home directory (`~/.spanbertcoref.conf`).

(**Optional**) To specify where to download pretrained models, modify `model_root` in the conf file.
By default, it is `${HOME}/.spanbertcoref/model/`.


4. Download pretrained models

```console
foo@bar:~$ python -m spanbertcoref.download spanbert_large
```

Specify which model to download. Your choice is: `spanbert_base`, `bert_base`, `spanbert_large`, or `bert_large`. I recommend `spanbert_large` for the best results (according to [the original repository](https://github.com/mandarjoshi90/coref)).

Pretrained models are larger than 1GB. Be prepared.


## Usage

Prepare input files with the following rules:

- One document per one file.
- One sentence per one line.


### Basic usage

```console
foo@bar:~$ cat input.txt
Dua Lipa (born 22 August 1995) is an English singer and songwriter.
After working as a model, she signed with Warner Bros.
Records in 2014 and released her eponymous debut album in 2017.
The album peaked at number three on the UK Albums Chart and yielded eight singles, including "Be the One", "IDGAF", and the UK number-one single "New Rules", which also peaked at number six in the US.

foo@bar:~$ spanbertcoref -m spanbert_large input.txt
```

The output is in a JSONL format.
Each line represents the result of coreference resolution for each input document.

```json
{
  "input_filename": "../test.txt",
  "clusters": {
    "0": [
      [[0, 0], [0, 7], "Dua Lipa"],
      [[1, 26], [1, 28], "she"],
      [[2, 29], [2, 31], "her"]
    ],
    "1": [
      [[2, 29], [2, 53], "her eponymous debut album"],
      [[3, 0], [3, 8], "The album"]
    ]
  },
  "annotated_text": [
    "<ent_0>Dua Lipa</ent_0> (born 22 August 1995) is an English singer and songwriter. After working as a model, <ent_0>she</ent_0> signed with Warner Bros.",
    "Records in 2014 and released <ent_1><ent_0>her</ent_0> eponymous debut album</ent_1> in 2017.",
    "<ent_1>The album</ent_1> peaked at number three on the UK Albums Chart and yielded eight singles, including \"Be the One\", \"IDGAF\", and the UK number-one single \"New Rules\", which also peaked at number six in the US."
  ]
}
```

- `input_filename`: The filename of the input document.
- `clusters`: Clusters of mentions coreferring the same entity. This is a dict, where the key is a cluster ID, and its corresponding value is the cluster of mentions. The cluster is a list of mentions. Each mention is represented by (i) a start offset, (ii) an end offset, and (iii) a surface form. The offset consists of a sentence number and character index (both zero-based) in the original input document. For example, `[[2, 29], [2, 31], "her"]` means *`her`, a span starting from the 29-th character in the third sentence (`[2, 29]`) to the 31-st character in the third sentence (`[2, 31]`)*.
- `annotated_text`: An input text annotated with coreference relations in a SGML format. Each mention is wrapped by `<ent_i>...</ent_i>`, where `i` is a cluster ID.


### Advanced usage

#### Store the result in a file

```console
foo@bar:~$ spanbertcoref -m spanbert_large input.txt > foo.jsonl
```

#### Multiple inputs

```console
foo@bar:~$ spanbertcoref -m spanbert_large input1.txt input2.txt
```

#### Use a different model

```console
foo@bar:~$ spanbertcoref -m spanbert_base input.txt
```

#### Specify genre of text for more accurate results

```console
foo@bar:~$ spanbertcoref -m spanbert_large -g wb input.txt
```

The genre can be:
- `bc` (Broadcast Conversation)
- `bn` (Broadcast News)
- `mz` (Magazine) 
- `nw` (Newswire) 
- `pt` (Pivot Corpus) 
- `tc` (Telephone Conversation) 
- `wb` (Web Text).

The default value is `nw`.
See https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf for further details.
