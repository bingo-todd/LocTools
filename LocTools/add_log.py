import numpy as np


def add_log(logger, feat_path, model_outputs, item_format='',
            is_flush=False):
    """ log model output to file io
    Args:
        logger: file object
        feat_path: file path of model input
        model_outputs: model output, 2-d numpy array, [n_frame, len_output]
        item_format: specify the format how model_outputs is logged, e.g., '.2f'
        is_flush: flush io to file 
    Returns:
        None
    """
    if not isinstance(model_outputs, np.ndarray):
        model_outputs = np.asarray(model_outputs)

    row_str_all = []
    for row in model_outputs:
        row_str = ' '.join(
            map(
                lambda x: ('{:'+item_format+'}').format(x),
                row))
        row_str_all.append(row_str)

    logger.write(f'{feat_path}: '+'; '.join(row_str_all))
    logger.write('\n')

    if is_flush:
        logger.flush()
