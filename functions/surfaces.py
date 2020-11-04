import numpy as np
import os, sys

import nibabel as nib
from nibabel.freesurfer import mghformat, MGHImage

from scipy.stats import trim_mean

"""
    CORTICAL PARCELLATION AND SURFACES
"""

def load_surf_data(left_surface_list, right_surface_list):
    """ Load individual surface data, and generate nxp matrix, where
        n = number of subjects and p = number of vertices in corresponding mask.

        ***NOTE: this assumes that the surfaces are in fsaverage5 space, in mgh format     ***
        ***   -if downsampled to fsaverage,fsaverage6,etc then adjust n_vert               ***
        ***   -if different format: CIFTI, GIFTI, then probably best to use something else!***

        Parameters
        ----------
        left_surface_list : List of surface files (.mgh) to load, including path to surface
        right_surface_list : Corresponding list of right hemipshere (rh) surface files to load, including path to surface

        Returns
        -------
        surf_data : numpy array (n_subjects, n_vertices)
        Vectorised surface data for all subjects"""


    n_vert = 10242  #number of vertices in fsaverage5 hemisphere

    surf_data = np.zeros((len(left_surface_list), n_vert*2))

    for it,filename in enumerate(left_surface_list):
        dat = mghformat.load(filename)
        surf_data[it,:n_vert] = np.squeeze(np.array(dat.get_fdata()))
        dat = mghformat.load(right_surface_list[it])
        surf_data[it,n_vert:] = np.squeeze(np.array(dat.get_fdata()))


    return surf_data


def parcellateSurface(data, zero_vector, parc='HCP', area=False):
    """  parcellate data into HCP labels

        data: 2D array, subject x vertexwise data (concatenated lh and rh)
        zero_vector: 1D array, vector of length n_vert*2 where 1 indicates a zeroed value
        parc: 'HCP' or 'cust250', parcellate scheme to use
        area: if area, then use total instead of mean and apply log transform

        returns:
        parcel_data: 1D vector, robust mean/sum of surface values in each parcel"""

    parcDir = '/home/gball/PROJECTS/brainAges/parcellations'

    # load up maps
    if parc=='HCP':
        rh_labels, _, _  = nib.freesurfer.io.read_annot(parcDir + '/rh.HCP-MMP1-fsaverage5-noHipp.annot', orig_ids=False)
        lh_labels, _, _  = nib.freesurfer.io.read_annot(parcDir + '/lh.HCP-MMP1-fsaverage5-noHipp.annot', orig_ids=False)

    elif parc=='cust250':
        rh_labels, _, _  = nib.freesurfer.io.read_annot(parcDir + '/rh.custom500-fsaverage5.annot', orig_ids=False)
        lh_labels, _, _  = nib.freesurfer.io.read_annot(parcDir + '/lh.custom500-fsaverage5.annot', orig_ids=False)

    else:
        print('ERROR: unknown parcellation scheme')
        sys.exit(1)

    tmp_labels = rh_labels + (max(lh_labels))  # rh is higher than lh
    tmp_labels[rh_labels==0]=0
    rh_labels = tmp_labels.copy()

    # ignore zero vertices
    lh_labels = lh_labels[zero_vector[:10242]==0]
    rh_labels = rh_labels[zero_vector[10242:]==0]

    # concatenate
    labels = np.concatenate((lh_labels, rh_labels))

    # generate parcellated data - taking the robust mean (5%-95%) within each parcel
    n_labels = len(np.unique(lh_labels)[1:]) + len(np.unique(rh_labels)[1:])
    n_subs = len(data)
    parcel_data = np.zeros((n_subs, n_labels))

    for s in np.arange(n_subs):
        subject_data = data[s,:]
        for n,k in enumerate(np.unique(labels)[1:]):
            label_data = subject_data[labels==k]
            label_data = label_data[~np.isnan(label_data)]
            if area is True:
                trimmed_mean = np.log(np.sum(label_data))
            else:
                trimmed_mean = trim_mean(label_data, 0.05)
            parcel_data[s,n] = trimmed_mean

    return parcel_data
