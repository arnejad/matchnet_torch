# This code is extracted and adapted from https://github.com/hanxf/matchnet/blob/master/models/feature_net.pbtxt
# Execute this code once to obtain the datasets.

import os
import numpy as np
import skimage as skimage
from skimage.io import imsave, imread

DATASETS = [ 'notredame','yosemite', 'liberty']


def GetPatchImage(patch_id, container_dir):
    """Returns a 64 x 64 patch with the given patch_id. Catch container images to
       reduce loading from disk.
    """
    # Define constants. Each container image is of size 1024x1024. It packs at
    # most 16 rows and 16 columns of 64x64 patches, arranged from left to right,
    # top to bottom.
    PATCHES_PER_IMAGE = 16 * 16
    PATCHES_PER_ROW = 16
    PATCH_SIZE = 64

    # Calculate the container index, the row and column index for the given
    # patch.
    container_idx, container_offset = divmod(patch_id, PATCHES_PER_IMAGE)
    row_idx, col_idx = divmod(container_offset, PATCHES_PER_ROW)

    # Read the container image if it is not cached.
    container_img = skimage.img_as_ubyte(imread('%s/patches%04d.bmp' % \
                (container_dir, container_idx), as_gray=True))

    # Extract the patch from the image and return.
    patch_image = container_img[ PATCH_SIZE * row_idx:PATCH_SIZE * (row_idx + 1), PATCH_SIZE * col_idx:PATCH_SIZE * (col_idx + 1)]
    return patch_image

# Static variables initialization for GetPatchImage.



def createDB(container_dir, info_file, interest_file, save_dir):
    # Read the 3Dpoint IDs from the info file.
    with open(info_file) as f:
        point_id = [int(line.split()[0]) for line in f]


    # Read the interest point from the interest file. The fields in each line
    # are: image_id, x, y, orientation, and scale. We parse all of them as float
    # even though image_id is integer.
    with open(interest_file) as f:
            interest = [[float(x) for x in line.split()] for line in f]

    total = len(interest)
    nameLen = len(str(total))
    processed = 0
    labels = np.array([])
    metadatas = np.array([])
    for i, metadata in enumerate(interest):
        labels = np.append(labels, point_id[i])
        metadatas = np.append(metadatas, metadata)
        img = GetPatchImage(i, container_dir)
        filename =  '0'*(nameLen-len(str(processed))) + str(processed)
        imsave(save_dir+'/'+filename+".png", img)
        # print(metadatas)
        processed += 1
        if processed % 1000 == 0:
            print(processed, '/', total)

    np.savetxt(save_dir+'/'+"labels.csv", labels, delimiter=",")
    np.savetxt(save_dir+'/'+"metadata.csv", metadatas, delimiter=",")


def main():

    for db in DATASETS:
        print("ceating dataset " + db)
        save_dir = '/data/p306627/DBs/matchnet/' + db
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        container_dir = '/home/p306627/codes/matchnet/data/phototour/' + db
        info_file = '/home/p306627/codes/matchnet/data/phototour/' + db + '/info.txt'
        interest_file = '/home/p306627/codes/matchnet/data/phototour/' + db + '/interest.txt'
        createDB(container_dir, info_file, interest_file, save_dir)


        
if __name__== "__main__":
   main()