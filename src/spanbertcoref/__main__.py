
import sys
import json
import argparse
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .lib import util
from .lib.bert import tokenization
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


def main():
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

    config = util.initialize_from_env(args.model)
    tok = get_tokenizer(config, args.model)
    model = util.get_model(config)

    with tf.Session() as session:
        model.restore(session)

        with maybe_file_handle(args.output) as output_file:
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

                clusters = get_clusters(lines, example)

                out = {
                    "input_filename": input_filename,
                    "clusters": clusters,
                    "annotated_text": markup(lines, clusters),
                }

                print(json.dumps(out), file=output_file)


def maybe_file_handle(f):
    return open(f) if f is not None else sys.stdout


def get_tokenizer(config, model_name):
    return tokenization.FullTokenizer(vocab_file=os.path.join(config["model_root"], model_name, "vocab.txt"), do_lower_case=False)


def parse_text(args, config, tokenizer, lines):
    max_segment = config["max_segment_len"]
    sentences = list()

    for sent_num, line in enumerate(lines):
        if line.strip() == "":
            continue

        tokens = tokenizer.tokenize(line)

        char_offset_end = 0

        for token in tokens:
            if len(sentences) == 0 or len(sentences[-1])+2+1 > max_segment:
                sentences.append(list())  # Create new sentence if it overflows.

            find_token = token[2:] if token.startswith("##") else token

            char_offset_start = line.index(find_token, char_offset_end)
            char_offset_end = char_offset_start + len(find_token) - 1

            assert find_token == line[char_offset_start:char_offset_end+1]
            
            sentences[-1].append((token, "-", sent_num, 0, (sent_num, char_offset_start, char_offset_end)))

    sentences = [list(zip(*sent)) for sent in sentences]

    def _flatten(_x):
        return [_j for _i in _x for _j in _i]

    return {
        'doc_key': args.genre,
        'sentences': [["[CLS]"] + list(sent[0]) + ["[SEP]"] for sent in sentences],
        'speakers': [["[SPL]"] + list(sent[1]) + ["[SPL]"] for sent in sentences],
        'sentence_map': _flatten([[sent[2][0]] + list(sent[2]) + [sent[2][-1]] for sent in sentences]),
        'subtoken_map': _flatten([[0] + list(sent[3]) + [0] for sent in sentences]),
        'char_map': _flatten([[(-1, -1, -1)] + list(sent[4]) + [(-1, -1, -1)] for sent in sentences]),
        'clusters': [],
    }


def get_clusters(lines, output):
    def _extract_cluster(cluster):
        for m1, m2 in sorted(cluster, key=lambda x: x[0]):
            (m1s, m1i, _), (m2s, _, m2j) = output["char_map"][m1], output["char_map"][m2]

            if m1s == m2s:
                mtext = lines[m1s][m1i:m2j+1]  # within-segment start/end.
            else:
                mtext = lines[m1s][m1i:] + lines[m2s][:m2j+1]

            yield (m1s, m1i), (m2s, m2j), mtext

    return {i: list(_extract_cluster(c))
            for i, c in enumerate(output['predicted_clusters'])}


def markup(lines, clusters):
    twin_before = [[""]*len(sent) for sent in lines]
    twin_after = [[""]*len(sent) for sent in lines]
    all_tags = list()

    for i, mentions in clusters.items():
        for m in mentions:
            all_tags.append((i, m))

    for c, ((m1s, m1i), (m2s, m2j), _) in sorted(all_tags, key=lambda x: len(x[1][2])):
        twin_before[m1s][m1i] = f"<ent_{c}>" + twin_before[m1s][m1i]
        twin_after[m2s][m2j] += f"</ent_{c}>"

    return ["".join([sb+s+sa
                     for s, sb, sa in zip(sent, sent_twin_bf, sent_twin_af)])
            for sent, sent_twin_bf, sent_twin_af in zip(lines, twin_before, twin_after)]


if __name__ == "__main__":
    main()
