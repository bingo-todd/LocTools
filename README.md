# LocTools
A handy toolbox for sound localization.
As long as raw model output is saved in text file, in the following format
```
input_file_path; frame0_output; frame1_output; ...
```
you can use this toolbox to 
- calcualte mse and correct percentage using different chunksize
- visualize model outputs
- calculate confusion matrix
- calculate the distribution of localiation result
- plot train curve of your model, if 'loss_record' is used as the variable name for loss
- a friendly interface for paralleling, eg. ```easy_parallel(func, tasks, n_worker)
- other functions that manipulate files eg. npz, csv

```shell
LocTools/
├── add_loc_log.py
├── cal_confuse_matrix.py
├── cal_hist.py
├── change_npz_key.py
├── change_npz_value.py
├── clean_model_dir.py
├── combine_csv.py
├── easy_parallel.py
├── __init__.py
├── load_log.py
├── plot_result_eg.py
├── plot_train_curve.py
└── send_email.py
```


