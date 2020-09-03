# Color Normalization'

# %%
import numpy as np
import histomicstk as htk


# %%
##Color Normalization##
def color_norm(inputimg):

    refimg=np.array([
    [0.41460361,  0.20837725,  0.13390153],
    [0.7672234 ,  0.84140212,  0.48046768],
    [0.48935888,  0.49861949, -0.86673017]])

    stain_unmixing_routine_params= {'stains': ['hematoxylin', 'eosin'],
'stain_unmixing_method': 'macenko_pca'}

    mask_img = np.dot(inputimg[...,:3], [0.299, 0.587, 0.114])
    mask_img=np.where(mask_img<=215,False,True)
    img_norm=htk.preprocessing.color_normalization.deconvolution_based_normalization(inputimg,W_target=refimg,stain_unmixing_routine_params=stain_unmixing_routine_params,mask_out=mask_img)
    return img_norm


