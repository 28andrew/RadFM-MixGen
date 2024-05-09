import csv
import glob
import re
import sys

import datasets
from nltk.tokenize import wordpunct_tokenize
import json
import logging
from vilmedic.blocks.scorers.scores import compute_scores


def process(impression):
    impression = impression.lower()
    return ' '.join(wordpunct_tokenize(impression))


def main(checkpoint_num):
    print(f'Eval for #{checkpoint_num}')
    files = glob.glob(f'../output/inference-mixed-finetuned-mixed-{checkpoint_num}.pt-rank*.csv')
    print(f'File count: {len(files)}')

    # Pred
    pred = {}
    for path in files:
        with open(path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                pred[int(row['idx'])] = row['prediction']

    print(f'Pred count: {len(pred)}')

    # Split pred into findings and predictions:
    pred_findings_d = {}
    pred_impressions_d = {}
    for idx, text in pred.items():
        match = re.search(r'Finding: (.*?)(?=\s*Impression:|$)', text, re.S)
        if match:
            findings = match.group(1).strip()
        else:
            findings = text

        match = re.search(r'Impression: (.*)', text, re.S)
        if match:
            impression = match.group(1).strip()
        else:
            impression = text

        pred_findings_d[idx] = findings
        pred_impressions_d[idx] = impression

    pred_findings = [pred_findings_d[key] for key in sorted(pred_findings_d.keys())]
    pred_impressions = [pred_impressions_d[key] for key in sorted(pred_impressions_d.keys())]

    # Ref
    print('Loading interpret-cxr-test-public dataset')
    dataset = datasets.load_dataset("StanfordAIMI/interpret-cxr-test-public")['test']
    findings_idxs = []
    ref_findings = []
    impressions_idxs = []
    ref_impressions = []

    i = 0
    for item in dataset:
        if item['findings']:
            findings_idxs.append(i)
            ref_findings.append(item['findings'])
        if item['impression']:
            impressions_idxs.append(i)
            ref_impressions.append(item['impression'])

        i += 1

    # Eval
    print(f'{len(findings_idxs)}, last few: {findings_idxs[-10:]}')
    print(f'pred_findings len: {len(pred_findings)}')
    pred_findings = [process(pred_findings[i]) for i in findings_idxs]
    pred_impressions = [process(pred_impressions[i]) for i in impressions_idxs]

    print('Computing findings metrics')
    print(json.dumps(compute_scores(["ROUGEL", "bertscore", "BLEU", "radgraph", "chexbert"],
                                    refs=ref_findings,
                                    hyps=pred_findings,
                                    split=None,
                                    seed=None,
                                    config=None,
                                    epoch=None,
                                    logger=logging.getLogger(__name__),
                                    dump=False),
                     indent=4)
          )

    print('Computing impressions metrics')
    print(json.dumps(compute_scores(["ROUGEL", "bertscore", "BLEU", "radgraph", "chexbert"],
                                    refs=ref_impressions,
                                    hyps=pred_impressions,
                                    split=None,
                                    seed=None,
                                    config=None,
                                    epoch=None,
                                    logger=logging.getLogger(__name__),
                                    dump=False),
                     indent=4)
          )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Provide checkpoint number')
        exit()

    main(int(sys.argv[1]))