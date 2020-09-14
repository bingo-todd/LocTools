import numpy as np


def add_loc_log(log_file, feat_file_path, model_outputs):
    if not isinstance(model_outputs, np.ndarray):
        model_outputs = np.asarray(model_outputs)

    outputs_str = '; '.join(map(lambda row: ' '.join(map(str, row)), 
                            model_outputs))
    log_file.write(f'{feat_file_path}; {outputs_str}\n')
    log_file.flush()
    
