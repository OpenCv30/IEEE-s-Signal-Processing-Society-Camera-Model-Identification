#https://bitbucket.org/polimi-ispl/camera-model-identification-with-cnn/src/d9cceca1c7501e0866cc4c6b9e4c620700943cb2/patch_extractor.py?at=master&fileviewer=file-view-default

def mid_intensity_high_texture(img):
    """
    :param img: 2D or 3D ndarray. Values are expected in [0,1] if img is float, in [0,255] if img is uint8
    :return score: score in [0,1]. Score tends to 1 as intensity is not saturated and high texture occurs
    """
    if img.dtype == np.uint8:
        img = img / 255.

    mean_std_weight = .7

    num_ch = 1 if img.ndim == 2 else img.shape[-1]
    img_flat = img.reshape(-1, num_ch)
    ch_mean = img_flat.mean(axis=0)
    ch_std = img_flat.std(axis=0)

    ch_mean_score = -4 * ch_mean ** 2 + 4 * ch_mean
    ch_std_score = 1 - np.exp(-2 * np.log(10) * ch_std)

    ch_mean_score_aggr = ch_mean_score.mean();ch_std_score_aggr = ch_std_score.mean()

    score = mean_std_weight * ch_mean_score_aggr + (1 - mean_std_weight) * ch_std_score_aggr
    return score
