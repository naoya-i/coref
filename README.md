# spanbert-coref

***Work in progress***

Coreference resolution system powered by BERT/SpanBERT. This is a thin wrapper for https://github.com/mandarjoshi90/coref.


## Setup

1. Create and activate a dedicated virtual environment


2. Install the software

```console
foo@bar:~$ pip install git+https://github.com/naoya-i/spanbert-coref
```

3. Run setup

```console
foo@bar:~$ python -m spanbertcoref.configure
```

This will compile a tensorflow kernel and copy a basic configuration file to your home directory (`~/.spanbertcoref.conf`).

(**Optional**) To specify where to download pretrained models, modify `model_root` in the conf file.
By default it is `${HOME}/.spanbertcoref/model/`.


4. Download pretrained models

```console
foo@bar:~$ python -m spanbertcoref.download spanbert_large
```

Your choice is: `spanbert_base`, `bert_base`, `spanbert_large`, or `bert_large`. I recommend `spanbert_large` for the best results (according to [the original repository](https://github.com/mandarjoshi90/coref)).

Typically, pretrained models are larger than 1GB. Be prepared.


## Usage

Prepare input files with the following rules:

- One document per one file.
- One sentence per one line.


### Basic usage

```console
foo@bar:~$ cat input.txt
Mount Fuji is the highest mountain in Japan, standing 3,776.24 m (12,389.2 ft).
It is the second-highest volcano located on an island in Asia (after Mount Kerinci on the island of Sumatra), and seventh-highest peak of an island on Earth.
Mount Fuji is an active stratovolcano that last erupted from 1707 to 1708.

foo@bar:~$ spanbertcoref -m spanbert_large input.txt
```

The output is in a JSONL format. Each line represents the result of coreference resolution for each input document.

```json
{"input_filename": "input.txt", "clusters": {"0": [[[0, 0], [0, 9], "Mount Fuji"], [[1, 0], [1, 1], "It"], [[2, 0], [2, 9], "Mount Fuji"]]}, "annotated_text": ["<ent_0>Mount Fuji</ent_0> is the highest mountain in Japan, standing 3,776.24 m (12,389.2 ft).", "<ent_0>It</ent_0> is the second-highest volcano located on an island in Asia (after Mount Kerinci on the island of Sumatra), and seventh-highest peak of an island on Earth.", "<ent_0>Mount Fuji</ent_0> is an active stratovolcano that last erupted from 1707 to 1708."]}
```

- `input_filename`: The filename of the input document.
- `clusters`: Clusters of mentions coreferring the same entity. This is a dict, where the key is a cluster name and its corresponding value is a cluster of mentions. The cluster is a list of mentions. Each mention is represented by (i) a start offset, (ii) an end offset, and a surface form. The offset consists of a sentence number and character index (both zero-based) in the original input document. For example, `[[1, 0], [1, 1], "It"]` means *a span starting from 0-th character to 1-st character in the second sentence*.
- `annotated_text`: An input text annotated with coreference relations in a SGML format.


### Advanced usage

#### Multiple inputs

```console
foo@bar:~$ spanbertcoref -m spanbert_large input1.txt input2.txt
```

#### Use a different model

```console
foo@bar:~$ spanbertcoref -m spanbert_base input1.txt input2.txt
```

#### Specify target genre

WIP

