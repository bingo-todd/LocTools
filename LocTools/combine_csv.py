import argparse
import pandas


def combine_csv(csv_path, csv_label, combined_csv_path):
    
    if not isinstance(csv_path, list):
        csv_paths = [csv_path]
        csv_labels = [csv_label]
    else:
        csv_paths = csv_path
        csv_labels = csv_label

    data_frame_tmp = pandas.read_csv(csv_paths[0])
    n_cur_label = len(['' for title in data_frame_tmp.columns if 'Label' in title])
    new_label_title = f'Label{n_cur_label}'

    data_frame_combined = pandas.DataFrame(data=None, columns=[new_label_title])
    for csv_path, label in zip(csv_paths, csv_labels):
        data_frame_combined = data_frame_combined.append(
            {new_label_title: label}, 
            ignore_index=True)
        data_frame_tmp = pandas.read_csv(csv_path)
        data_frame_combined = data_frame_combined.append(
            data_frame_tmp,
            ignore_index=True)
    data_frame_combined.to_csv(combined_csv_path, index=False, index_label=None)


def parse_args():
    parser = argparse.ArgumentParser(description='combine csv files into one')
    parser.add_argument('--combined-csv-path', dest='combined_csv_path',
                        required=True, type=str,
                        help='where combined scv files saved')
    parser.add_argument('--csv-path', dest='csv_path',
                        required=True, type=str, nargs='+', 
                        help='csv file to be combined')
    parser.add_argument('--csv-label', dest='csv_label', nargs='+',
                        required=True, type=str, 
                        help='label of csv file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    combine_csv(args.csv_path, args.csv_label, args.combined_csv_path)


if __name__ == '__main__':
    main()
