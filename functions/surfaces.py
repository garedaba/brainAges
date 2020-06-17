import numpy as np
import os, sys

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


def parcellateSurface(data, zero_vector, parc='HCP'):
    """  parcellate data into HCP labels

        data: 2D array, subject x vertexwise data (concatenated lh and rh)
        zero_vector: 1D array, vector of length n_vert*2 where 1 indicates a zeroed value
        parc: 'HCP' or 'cust250', parcellate scheme to use

        returns:
        parcel_data: 1D vector, robust mean of surface values in each parcel"""

    parcDir = '/home/gball/PROJECTS/brainAges/parcellations'

    # load up maps
    if parc=='HCP':
        rh_labels = mghformat.load(parcDir + '/rh.HCP-MMP1.mgh')
        rh_labels = np.array(rh_labels.get_fdata()).squeeze()

        lh_labels  = mghformat.load(parcDir + '/lh.HCP-MMP1.mgh')
        lh_labels = np.array(lh_labels.get_fdata()).squeeze()

    elif parc=='cust250':
        rh_labels = mghformat.load(parcDir + '/rh.custom500.mgh')
        rh_labels = np.array(rh_labels.get_fdata()).squeeze()

        lh_labels  = mghformat.load(parcDir + '/lh.custom500.mgh')
        lh_labels = np.array(lh_labels.get_fdata()).squeeze()

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
            trimmed_mean = trim_mean(label_data, 0.05)
            parcel_data[s,n] = trimmed_mean

    return parcel_data


def save_surface_out(data, plot_path, hemi='lh', parc='HCP', template='fsaverage5'):
        """save data to a freesurfer surface file.
        can be parcellated ('HCP', 'cust250') or vertexwise ('vertex')

        data, 1D array, data to be saved
        plot_path, path to output directory
        hemi: 'lh' or 'rh'
        parc: 'HCP', 'cust250', 'vertex'
        template: 'fsaverage' or 'fsaverage5'

        """
        parc_dir = '/home/gball/PROJECTS/brainAges/parcellations'

        if parc=='HCP':
                labels  = mghformat.load(parc_dir + '/' + hemi + '.HCP-MMP1.mgh')
                labels = np.array(labels.get_fdata()).squeeze()

                out_data = np.zeros(np.shape(labels))
                for n,i in enumerate(np.arange(max(labels))):
                        out_data[labels==n+1] = data[n]

        elif parc=='cust250':
                labels  = mghformat.load(parc_dir + '/' + hemi + '.custom500.mgh')
                labels = np.array(labels.get_fdata()).squeeze()

                out_data = np.zeros(np.shape(labels))
                for n,i in enumerate(np.arange(max(labels))):
                        out_data[labels==n+1] = data[n]

        elif parc=='vertex':
                out_data = data

        else:
            print('ERROR: unknown parcellation scheme')
            sys.exit(1)

        surfname =os.environ['FREESURFER_HOME'] + '/subjects/' + template + '/surf/' + hemi + '.white.avg.area.mgh'
        surf_data = mghformat.load(surfname)
        surf_data.get_fdata()[:] = out_data.reshape(-1,1,1)

        comp=MGHImage((np.asarray(surf_data.get_fdata())),
                                        surf_data.affine ,
                                        extra=surf_data.extra,
                                        header=surf_data.header,
                                        file_map=surf_data.file_map)
        mghformat.save(comp, plot_path + '/' + hemi + '.out_data.mgh')
