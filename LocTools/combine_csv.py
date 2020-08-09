import sys
import pandas


def combine_csv(combined_csv_path, *csv_file_label):

    n_file = int(len(csv_file_label)/2)

    data_frame_tmp = pandas.read_csv(csv_file_label[0])
    n_cur_label = len(['' for title in data_frame_tmp.columns
                       if 'Label' in title])
    new_label_title = f'Label{n_cur_label}'

    data_frame_combined = pandas.DataFrame(data=None,
                                           columns=[new_label_title])
    for file_i in range(n_file):
        data_frame_combined = data_frame_combined.append(
            {new_label_title: csv_file_label[file_i*2+1]},
            ignore_index=True)
        data_frame_tmp = pandas.read_csv(csv_file_label[file_i*2])
        data_frame_combined = data_frame_combined.append(
            data_frame_tmp,
            ignore_index=True)

    data_frame_combined.to_csv(combined_csv_path, index=False,
                               index_label=None)


if __name__ == '__main__':
    combine_csv(*sys.argv[1:])
