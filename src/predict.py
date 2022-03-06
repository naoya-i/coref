from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import argparse
import os
import logging

import tensorflow as tf
from lib import util
from lib.bert import tokenization
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(args):
    config = util.initialize_from_env(args.model)
    tok = get_tokenizer(args.model)

    # with open(args.inputs[0]) as fp:
    #     example = parse_text(args, config, tok, fp.read())
    #     print(example)
    #     return

    model = util.get_model(config)

    with tf.Session() as session:
        model.restore(session)

        with get_file_handle(args.output) as output_file:
            for example_num, input_filename in enumerate(tqdm(args.inputs)):
                with open(input_filename) as fp:
                    lines = fp.read().split("\n")
                    example = parse_text(args, config, tok, lines)
                
                tensorized_example = model.tensorize_example(example, is_training=False)
                feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}

                _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)

                predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
                example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
                example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
                example['head_scores'] = []

                print(json.dumps(parse_output(lines, example)), file=output_file)


def get_file_handle(f):
    return open(f) if f is not None else sys.stdout


def get_tokenizer(model_name):
    return tokenization.FullTokenizer(vocab_file=os.path.join("model", model_name, "vocab.txt"), do_lower_case=False)


def parse_output(lines, output):
    clusters, mentions = list(), list()

    def convert_mention(mention):
        assert output["char_map"][mention[0]][0] == output["char_map"][mention[1]][0]
        nmention = (output["char_map"][mention[0]][0], output["char_map"][mention[0]][1], output["char_map"][mention[1]][2])
        mtext = lines[nmention[0]][nmention[1]:nmention[2]+1]
        return (nmention, mtext)

    for cluster in output['predicted_clusters']:
        mapped = []

        for mention in cluster:
            mapped.append(convert_mention(mention))
        
        clusters.append(mapped)


    for mention in output['top_spans']:
        mentions.append(convert_mention(mention))

    return {
        "mentions": mentions,
        "clusters": clusters,
    }


def parse_text(args, config, tokenizer, lines):
    """
    Credit: https://colab.research.google.com/drive/1SlERO9Uc9541qv6yH26LJz5IM9j7YVra#scrollTo=H0xPknceFORt
    """
    max_segment = config["max_segment_len"]

    data = {
        'doc_key': args.genre,
        'sentences': [["[CLS]"]],
        'speakers': [["[SPL]"]],
        'clusters': [],
        'sentence_map': [0],
        'subtoken_map': [0],
        'char_map': [(-1, -1)],
    }

    subtoken_num = 0

    for sent_num, line in enumerate(lines):
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
            data['char_map'].append((-1, -1, -1))

        char_offset_end = 0
        ctoken = raw_tokens[0]
        cpos = 0

        for token in tokens:
            data['sentences'][-1].append(token)
            data['speakers'][-1].append("-")
            data['sentence_map'].append(sent_num)
            data['subtoken_map'].append(subtoken_num)

            if token.startswith("##"):
                token = token[2:]

            char_offset_start = line.index(token, char_offset_end)
            char_offset_end = char_offset_start + len(token) - 1

            data['char_map'].append((sent_num, char_offset_start, char_offset_end))

            assert token == line[char_offset_start:char_offset_end+1]
            
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
    data['char_map'].append((-1, -1, -1))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--genre', default="nw",
        help="Genre of input document.")
    parser.add_argument(
        '-m', '--model', default="spanbert_base", choices="bert_base spanbert_base bert_large spanbert_large".split(),
        help="Model type.")
    parser.add_argument(
        '-o', '--output',
        help="Output path.")
    parser.add_argument(
        'inputs', nargs="+",
        help="Input document(s).")
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    main(args)
