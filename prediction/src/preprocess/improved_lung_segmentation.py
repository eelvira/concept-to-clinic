import numpy as np
import scipy.ndimage
import skimage
import skimage.measure
import skimage.filters
import skimage.morphology
import scipy.ndimage.morphology
import skimage.transform


def region_growing(img, seed, minthr, maxthr, structure=None):
    """
    code was taken from:
    https://github.com/loli/medpy/wiki/Basic-image-manipulation
    """
    img[seed] = minthr
    thrimg = (img < maxthr) & (img >= minthr)
    lmap, _ = scipy.ndimage.label(thrimg, structure=structure)
    lids = np.unique(lmap[seed])
    region = np.zeros(img.shape, np.bool)
    for lid in lids:
        region |= lmap == lid
    return region


def extract_bronchial(ct_slice, xy_spacing=1., BRONCHIAL_THRESHOLD=-950):
    """Detection the bronchi and the trachea on CT.

    The bronchi and the trachea have the following properties:
        (1) an average HU below −950,
        (2) a minimum size of 50 mm 2 ,
        (3) a maximum size of 1225 mm 2 ,
        (4) mean x- and y-coordinates not further than 30% of the x- and
        y-dimensions of the image away from the center of the slice.

    Args:
        ct_slice;
        xy_spacing (float);
        BRONCHIAL_THRESHOLD (int): an average HU in the bronchi.

    Returns:
        np.ndarray[bool]: a label of the segmented bronchi and trachea on CT

    """
    labeled = skimage.measure.label(ct_slice < BRONCHIAL_THRESHOLD)
    areas = np.bincount(labeled.flatten())
    labels = [i
              for i, area in enumerate(areas)
              if (area * xy_spacing >= 50) and (area * xy_spacing <= 1225)]
    coords = [np.where(labeled == i) for i in labels]

    center = np.array(ct_slice.shape) // 2
    max_dist = np.array(ct_slice.shape) * .3
    labels = [(np.mean(coord, axis=1), labe)
              for labe, coord in zip(labels, coords)
              if (abs(center - np.mean(coord, axis=1)) < max_dist).all()]

    if len(labels) != 0:
        return labeled == min(labels, key=lambda x: sum((x[0] - center) ** 2))[1]

    return None


def select_bronchial(bronchial, ct_slices, levels):
    """ Selection bronchi.

    Args:
        bronchial: a label of the pixels with the bronchi;
        ct_slices: a label of CT;
        levels: a number of slice with the bronchi.

    """
    center = np.array(bronchial[0].shape) // 2
    coords = [(np.mean(np.where(bool_slice), axis=1), i)
              for i, bool_slice in enumerate(bronchial)]
    el = min(coords, key=lambda x: sum((x[0] - center) ** 2))
    return bronchial[el[1]], ct_slices[el[1]], levels[el[1]]


def select_seeds(bronchial, ct_clice):
    """Selection initial pixel.

    Args:
        bronchial: a label of the pixels with the bronchi;
        ct_clice:

    Returns:
        min pixel in area of bronchi

    """
    return (ct_clice * bronchial) == ct_clice[bronchial].min()


def extract_seeds(patient):
    """ Extraction initial region.

    Starting at the top, the first 25 slices are examined for
    suitable regions.

    Args:
        patient (lnp.ndarray[bool]): a label of CT;

    Returns:
        np.ndarray[bool]: seeds - a initial region.

    """
    bronchials = list()
    bronch_cts = list()
    levels = list()
    for i in range(25):
        bronchial = extract_bronchial(patient[i])
        if bronchial is not None:
            bronchials.append(bronchial)
            bronch_cts.append(patient[i])
            levels.append(i)

    for i in range(-25, 0, 1):
        bronchial = extract_bronchial(patient[i])
        if bronchial is not None:
            bronchials.append(bronchial)
            bronch_cts.append(patient[i])
            levels.append(i)

    bronchial, ct_slice, level = select_bronchial(bronchials,
                                                  bronch_cts,
                                                  levels)

    seeds = np.zeros(patient.shape)
    seeds[level] = select_seeds(bronchial, ct_slice)

    return seeds


def growing_bronchis(patient, seeds,
                     threshold=-950,
                     step=64,
                     full_extraction=True):
    """Extraction of large airways.

    Before the lungs are segmented, the trachea and
    the bronchi are extracted using the technique of a region growing.
    To initialize the region growing, the start point is automatically
    identified on the upper axial scan slices by searching for related
    areas with the specified properties.
    When an explosion occurred, the increase in threshold was divided by 2.

    Args:
        patient: a label of CT;
        seeds: a initial region (pixel);
        threshold: an average HU;
        step: In each iteration, the threshold was initially increased with 64 HU
        full_extraction: full extraction of large airways.

    Returns:
        ret - a label of the bronchi at the previous stage.
        seeds - a label of the bronchi.

    """

    seeds = seeds.astype(np.bool_)
    seeds = region_growing(patient.copy(), seeds, -5010, threshold)
    volume = np.count_nonzero(seeds)

    lungs_thresh = skimage.filters.threshold_otsu(patient[patient.shape[0] // 2])

    ret = None
    while True:
        labeled = region_growing(patient.copy(), seeds, -5010, threshold + step)
        new_volume = np.count_nonzero(labeled)
        if new_volume >= volume * 2:
            if step == 4:
                ret = seeds.copy()
                if not full_extraction:
                    return ret

            if step == 2:
                return ret, seeds
            step = np.ceil(step * 0.5)
            continue

        threshold += step
        volume = new_volume
        seeds = labeled

        if threshold >= lungs_thresh:
            if ret is None:
                ret = seeds.copy()

            if not full_extraction:
                return ret

            return ret, seeds


def grow_lungs(patient, seeds):
    """ Segmentation of lung regions.

    The lungs are segmented using region growing. As a
    seed point for the region growing, the voxel with the lowest
    HU within the airways is used.

    To determine the upper threshold for the
    region growing operation, optimal thresholding is applied.

    Args:
        patient (np.ndarray[bool]): a label of CT

    Returns:
        np.ndarray[bool]: a label of the segmented lungs.

    """
    lungs_seeds = patient * seeds == patient[seeds].min()
    lungs_seeds = lungs_seeds.astype(np.bool_)
    threshold = skimage.filters.threshold_otsu(patient[patient.shape[0] // 2])

    lungs_seeds = region_growing(patient.copy(), lungs_seeds, -1010, threshold)
    return scipy.ndimage.morphology.binary_opening(lungs_seeds - scipy.ndimage.morphology.binary_opening(seeds))


def lungs_postprocessing(lungs_seeds):
    """ Smoothing.

    First 3D hole filling is applied
    to include vessels and other high-density structures that were
    excluded by the threshold used in region growing, in the segmentation.

    Next, morphological closing with a spherical
    kernel is applied. The diameter of the kernel is set to 2% of
    the x-dimension of the image.

    Args:
        lungs_seeds (np.ndarray[bool]): a label with the segmented lungs before postprocessing

    Returns:
        np.ndarray[bool]: a label with the segmented lungs after binary_fill_holes

    """
    for i in range(lungs_seeds.shape[1]):
        lungs_seeds[:, i] = scipy.ndimage.morphology.binary_fill_holes(lungs_seeds[:, i])

    return lungs_seeds


def flip_lung(patient, trachea):
    """ Аlipping of the lungs.

    If no suitable region is detected  in the top slices,
    the bottom slices of the scan are inspected  to be able
    to handle cases that were scanned in a reverse  direction.

    Args:
        patient(np.ndarray[bool]): a label of CT
        trachea (np.ndarray[bool]): a label of the trachea

    Returns:
        np.ndarray[bool]: patient - converted Lungs
    """
    z_coord, y_coord, x_coord = np.where(trachea)
    z, y, x = np.where(patient)

    z = np.unique(z)
    z_mean = np.mean(np.unique(z_coord), axis=0)
    z_down = abs(z[0] - z_mean)
    z_up = abs(z[-1] - z_mean)
    if z_down > z_up:
        patient = np.flip(patient, axis=0)
    return patient


def conventional_lung_segmentation(patient):
    """ The lungs segmentation.

    The algorithm consists of the following steps:
        (1)Extraction of large airways;
        (2)Segmentation of lung regions;
        (3)Separation of the left and right lungs;
        (4)Smoothing.

    The function aggregates extract_seeds, growing_bronchis,
    flip_lung, grow_lungs, lungs_postprocessing.

    Args:
        patient (np.ndarray[bool]): a label of CT

    Returns:
        np.ndarray[bool]: lungs_seeds - a label with the segmented lungs.
        np.ndarray[bool]: trachea - a label with the segmented trachea.
    """

    seeds = extract_seeds(patient)
    trachea, _ = growing_bronchis(patient, seeds)
#     patient = flip_lung(patient, trachea)

#     seeds = extract_seeds(patient)
#     trachea, _ = growing_bronchis(patient, seeds)

    lungs_seeds = grow_lungs(patient, trachea)

    selem = skimage.morphology.ball(int(patient.shape[-1] * .01))
    lungs_seeds = skimage.morphology.binary_closing(lungs_seeds, selem)

    lungs_seeds = lungs_postprocessing(lungs_seeds)

    return lungs_seeds, trachea


def cumulation(lung):
    """The cumulative x-position inside a lung.

    Args:
        lung: The segmented lung (left or right)

    Returns:
        lung: cumulative x-position
        z_coords: z-coordinates in lung

    """
    z_coords, y_coords, x_coords = np.where(lung)
    current_volume = 0
    volume = np.count_nonzero(lung)
    for coord in np.unique(x_coords):
        current_volume += np.count_nonzero(lung[:, :, coord])
        lung[:, :, coord][lung[:, :, coord] != 0] = current_volume / float(volume)
    return lung, z_coords


def ventricular_extraction(err_lung):
    """ The extraction of the ventricle.

    On all CT there is a ventricle. In order for the algorithm
    doesn't define the ventricle as a segmentation error,
    it is necessary to remove the error mask of the largest
    size that corresponds to the ventricle.

    Args:
        err_lung (np.ndarray[bool]): the label of errors in the lung.

    Returns:
        np.ndarray[bool]: err_lung - the label of errors in the lung without the ventricle.

    """
    max_marker = -1
    z_max = -1
    lav, mar = scipy.ndimage.label(err_lung)
    volumes = np.hstack([np.bincount(lav.flatten()), [-1]])

    for i in range(1, mar + 1):
        z, y, x = np.where(lav == i)
        curr_z = z.max()
        if curr_z > z_max:
            curr_z = z_max
            coords = (z, y, x)

        if curr_z == z_max:
            if volumes[i] > volumes[max_marker]:
                # curr_z = z_max
                coords = (z, y, x)

        err_lung[coords] = 0
    return err_lung


def costal_surface(lung, z_coords, max_coor, combined):
    """ The costal surface.

    The convexity is determined by comparing the costal lung surface
    in axial slices to the convex hull of this costal lung surface.

    Args:
        lung: cumulative x-position in lung
        z_coords: z-coordinates in lung
        max_coor: threshold for the сostal surface for the lung
        combined (np.ndarray[bool]): segmented lung (left or right)

    """
    erroneus = np.zeros(lung.shape)
    for coord in np.unique(z_coords):
        if max_coor == .8:
            ROI = (lung[coord] < max_coor) * (lung[coord] != 0)
        else:
            ROI = (lung[coord] > max_coor) * (lung[coord] != 0)

        erroneus[coord] = skimage.morphology.convex_hull_object(ROI) - ROI

    erroneus = erroneus * (1 - (combined != 0))
    erroneus_erosion = scipy.ndimage.morphology.binary_erosion(erroneus, structure=skimage.morphology.ball(4.5))
    erroneus_erosion = scipy.ndimage.morphology.binary_dilation(erroneus_erosion,
                                                                structure=skimage.morphology.ball(4.5))

    return lung, erroneus, erroneus_erosion


def detection_lung_error(patient):
    """ Automatic error detection.

    The function aggregates cumulation, costal_surface, ventricular_extraction.
    Args:
        patient

    Returns:
        lung_l - the mask of left lung
        lung_r - the mask of right lung
        trachea - the mask of trachea

    """

    combined, trachea = conventional_lung_segmentation(patient)

    lung_right = ((combined.copy() == 1)
                  | (combined.copy() == 3)
                  | (combined.copy() == 5)
                  | (combined.copy() == 7)
                  | (combined.copy() == 9)
                  | (combined.copy() == 11)
                  | (combined.copy() == 13)
                  | (combined.copy() == 15))

    lung_left = ((combined.copy() == 2)
                 | (combined.copy() == 4)
                 | (combined.copy() == 6)
                 | (combined.copy() == 8)
                 | (combined.copy() == 10)
                 | (combined.copy() == 12)
                 | (combined.copy() == 14))

    lung_right_c = lung_right.astype(float)
    lung_left_c = lung_left.astype(float)

    lung_right_comm, z_coords_r = cumulation(lung_right_c)
    lung_left_comm, z_coords_l = cumulation(lung_left_c)

    lung_left_1, er_l, erroneus_erosion_left = costal_surface(lung_left_comm, z_coords_l, .2, lung_left_c)
    lung_right_1, er_r, erroneus_erosion_right = costal_surface(lung_right_comm, z_coords_r, .8, lung_right_c)

    erroneus_erosion_left = ventricular_extraction(erroneus_erosion_left.copy())
    erroneus_erosion_right = ventricular_extraction(erroneus_erosion_right.copy())
    erroneus_erosion_left = ventricular_extraction(erroneus_erosion_left)
    erroneus_erosion_right = ventricular_extraction(erroneus_erosion_right)

    lung_l = lung_left.copy() + erroneus_erosion_left
    lung_r = lung_right.copy() + erroneus_erosion_right
    lungs = lung_l.copy() + lung_r.copy()
    
    return lungs, trachea
