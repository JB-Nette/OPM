#!/usr/bin/env python

'''
OPM post-processing using numpy, numba, skimage, and npy2bdv.
Modified version for UTSW OPM data
Orthgonal interpolation method as described by Vincent Maioli (http://doi.org/10.25560/68022)

Shepherd 09/20
'''

# imports
import numpy as np
from pathlib import Path
from natsort import natsorted, ns
import npy2bdv
import gc
import sys
import getopt
import re
import skimage.io as io
from tifffile import imsave
import os


from skimage.measure import block_reduce
import time
from numba import njit, prange

def write_tiff(output_path, filename, deskewed):
    print('Save deskewed data', output_path + '/' + filename)
    return imsave(output_path + '/' + filename, deskewed)

# perform OPM reconstruction using orthogonal interpolation
# http://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel
@njit(parallel=True)
def stage_deskew(data, parameters):
    # unwrap parameters
    theta = parameters[0]  # (degrees)
    distance = parameters[1]  # (nm)
    pixel_size = parameters[2]  # (nm)
    [num_images, ny, nx] = data.shape  # (pixels)
    print(data.shape)

    # change step size from physical space (nm) to camera space (pixels)
    pixel_step = distance / pixel_size  # (pixels)
    print("pixel_step", pixel_step)
    # calculate the number of pixels scanned during stage scan
    scan_end = num_images * pixel_step  # (pixels)
    print("scan end", scan_end)
    # calculate properties for final image
    final_ny = np.int64(np.ceil(scan_end + ny * np.cos(theta * np.pi / 180)))  # (pixels)
    final_nz = np.int64(np.ceil(ny * np.sin(theta * np.pi / 180)))  # (pixels)
    final_nx = np.int64(nx)  # (pixels)
    print("final_ny ", final_ny)
    print("final_nx ", final_nx)
    print("final_nz ", final_nz)

    # create final image
    output = np.zeros((final_nz, final_ny, final_nx), dtype=np.float32)  # (pixels,pixels,pixels - data is float32)

    # precalculate trig functions for scan angle
    tantheta = np.float32(np.tan(theta * np.pi / 180))  # (float32)
    sintheta = np.float32(np.sin(theta * np.pi / 180))  # (float32)
    costheta = np.float32(np.cos(theta * np.pi / 180))  # (float32)

    # perform orthogonal interpolation

    # loop through output z planes
    # defined as parallel loop in numba
    # http://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel
    for z in prange(0, final_nz):
        # calculate range of output y pixels to populate
        y_range_min = np.minimum(0, np.int64(np.floor(np.float32(z) / tantheta)))
        y_range_max = np.maximum(final_ny, np.int64(np.ceil(scan_end + np.float32(z) / tantheta + 1)))

        # loop through final y pixels
        # defined as parallel loop in numba
        # http://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel
        for y in prange(y_range_min, y_range_max):

            # find the virtual tilted plane that intersects the interpolated plane
            virtual_plane = y - z / tantheta

            # find raw data planes that surround the virtual plane
            plane_before = np.int64(np.floor(virtual_plane / pixel_step))
            plane_after = np.int64(plane_before + 1)

            # continue if raw data planes are within the data range
            if ((plane_before >= 0) and (plane_after < num_images)):

                # find distance of a point on the  interpolated plane to plane_before and plane_after
                l_before = virtual_plane - plane_before * pixel_step
                l_after = pixel_step - l_before

                # determine location of a point along the interpolated plane
                za = z / sintheta
                virtual_pos_before = za + l_before * costheta
                virtual_pos_after = za - l_after * costheta

                # determine nearest data points to interpoloated point in raw data
                pos_before = np.int64(np.floor(virtual_pos_before))
                pos_after = np.int64(np.floor(virtual_pos_after))

                # continue if within data bounds
                if ((pos_before >= 0) and (pos_after >= 0) and (pos_before < ny - 1) and (pos_after < ny - 1)):
                    # determine points surrounding interpolated point on the virtual plane
                    dz_before = virtual_pos_before - pos_before
                    dz_after = virtual_pos_after - pos_after

                    # compute final image plane using orthogonal interpolation
                    output[z, y, :] = (l_before * dz_after * data[plane_after, pos_after + 1, :] +
                                       l_before * (1 - dz_after) * data[plane_after, pos_after, :] +
                                       l_after * dz_before * data[plane_before, pos_before + 1, :] +
                                       l_after * (1 - dz_before) * data[plane_before, pos_before, :]) / pixel_step
    # return output
    return output


# parse experimental directory, load data, perform orthogonal deskew, and save as BDV H5 file
def main(argv):
    # parse directory name from command line argument
    input_dir_string = 'Y:/lightsheet stuff/20210406 Deskew of dot pattern_argolight'
    output_dir_string = 'Y:/lightsheet stuff/20210406 Deskew of dot pattern_argolight'

    try:
        arguments, values = getopt.getopt(argv, "hi:o:n:c:", ["help", "ipath=", "opath="])
    except getopt.GetoptError:
        print('Error. stage_recon.py -i <inputdirectory> -o <outputdirectory>')
        sys.exit(2)
    for current_argument, current_value in arguments:
        if current_argument == '-h':
            print('Usage. stage_recon.py -i <inputdirectory> -o <outputdirectory>')
            sys.exit()
        elif current_argument in ("-i", "--ipath"):
            input_dir_string = current_value
        elif current_argument in ("-o", "--opath"):
            output_dir_string = current_value

    if (input_dir_string == ''):
        print('Input parse error.')
        sys.exit(2)

    # Load data

    # https://docs.python.org/3/library/pathlib.html
    # Create Path object to directory
    input_dir_path = Path(input_dir_string)

    # Parse directory for number of channels and strip positions then sort
    sub_dirs = [x for x in input_dir_path.iterdir() if x.is_dir()]
    sub_dirs = natsorted(sub_dirs, alg=ns.PATH)

    # TO DO: automatically determine number of channels and tile positions
    # I think we don't need this one because we are not saving in .Hdf5 file
    # num_channels = 1
    # num_tiles = 2
    # one folder is 1 tile of y scan -> ch0_y0, ch0_y1

    # create parameter array
    # [theta, stage move distance, camera pixel size]
    # distance: step distance -JB
    # units are [degrees,nm,nm]
    params = np.array([29, 200, 122], dtype=np.float32)

    # check if user provided output path
    if (output_dir_string == ''):
        output_dir_path = input_dir_path
    else:
        output_dir_path = Path(output_dir_string)

    # https://github.com/nvladimus/npy2bdv
    # create BDV H5 file with sub-sampling for BigStitcher
    # output_path = output_dir_path / 'deskewed_10002.h5'
    # bdv_writer = npy2bdv.BdvWriter(str(output_path), nchannels=num_channels, ntiles=num_tiles,
    # subsamp=((1,1,1),),blockdim=((4, 256, 256),))
    # I dont' think we need this too -JB

    # I made a huge modification from this part -JB
    # We need to know the number of channels and z position and y position because the total number of sub folders relate to channel*z_num*y_num -JB
    # We can do it as user input or auto by changing the folder name (like what we do in Confocal system) - JB
    # loop over each directory. Each directory will be placed as a "tile" into the BigStitcher file

    for sub_dir in sub_dirs:

        print("Now processing ", str(sub_dir))

        # find all individual tif files in the current channel + tile sub directory and sort + deskew each file in sub directory
        JB_before_deskew(sub_dir=sub_dir, parameters=params, output_dir=output_dir_string)




def JB_before_deskew(sub_dir, parameters, output_dir):

    files = natsorted(sub_dir.glob('*.tif'), alg=ns.PATH)
    print('Deskewing data...')

    if len(files) == 1:
        stack = stack[0, :, :, :]
        deskewed = stage_deskew(stack, parameters)
        write_tiff(output_path = output_dir, filename=file_str[-1], deskewed=deskewed)

        #free up memmory
        del deskewed
        gc.collect()


    # check if we have more than 1 tiff file in 1 folder
    num_tiff_in_dir = len(files)
    print("number of tiff file in sub directory", num_tiff_in_dir)

    if num_tiff_in_dir > 1:
         for file in files:
            stack = np.asarray(io.imread(str(file)), dtype=np.float32)
            [num_image, nx, ny] = stack.shape
            deskewed = stage_deskew(stack, parameters)
            file_str = str(file).split("\\")
            write_tiff(output_path = output_dir, filename=file_str[-1], deskewed=deskewed)

            del deskewed
            gc.collect()

# run
if __name__ == "__main__":
    main(sys.argv[1:])

# The MIT License
#
# Copyright (c) 2020 Douglas Shepherd, Arizona State University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

