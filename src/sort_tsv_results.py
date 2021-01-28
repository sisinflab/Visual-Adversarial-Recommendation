import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run tsv results sorting.")
    parser.add_argument('--dataset', nargs='?', default='tradesy_original', help='amazon_men, amazon_women, amazon_sport')
    parser.add_argument('--metric', nargs='?', default='cndcg', help='chr, cndcg')
    parser.add_argument('--model', nargs='?', default='LFM', help='ACF, DVBPR')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.dataset == 'amazon_men':
        origin = 774
    elif args.dataset == 'amazon_women':
        origin = 610
    elif args.dataset == 'tradesy_original':
        origin = 834
    else:
        raise NotImplementedError('This dataset is not available!')

    list_regex = [
        "^original_top*",
        "^fgsm_*",
        "^pgd_*",
        "^cw_*",
        "^madry_original*",
        "^madry_fgsm*",
        "^madry_pgd*",
        "^madry_cw*",
        "^free_adv_original*",
        "^free_adv_fgsm*",
        "^free_adv_pgd*",
        "^free_adv_cw*",
    ]

    index_sorted = []

    filename = '../{0}/{1}/df_{2}_{3}.csv'.format(args.metric, args.dataset, args.metric, args.model)
    df = pd.read_csv(filename)
    df['p-value'] = df['p-value'].fillna('')
    df = df[df['classId'] == origin].reset_index(drop=True)
    df['score'] = df['score'].round(4)

    for reg in list_regex:
        try:
            index_sorted.append(
                df[df['experiment'].str.contains(reg)].index.values[0]
            )
        except IndexError:
            df = df.append({
                'top-k': '---',
                'experiment': reg[1:-1],
                'classId': '---',
                'className': '---',
                'position': '---',
                'score': '---',
                'p-value': '---'
            }, ignore_index=True)
            index_sorted.append(
                df[df['experiment'].str.contains(reg)].index.values[0]
            )

    df = df.reindex(index_sorted).reset_index(drop=True)

    for i, row in df.iterrows():
        if row['p-value'] == '*':
            df.loc[i, 'score'] = str(row['score']) + '^*'

    df.to_csv('../{0}/{1}/df_{2}_{3}_sorted.csv'.format(args.metric, args.dataset, args.metric, args.model), index=False)
