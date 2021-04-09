import time
import sys
import os
import numpy as np
import npy2bdv

print("Example1: writing 2 time points, 2 channels, 2 illuminations, 2 angles")
#plane = generate_test_image((1024, 2048))
#stack = []
#for z in range(50):
    #stack.append(plane)
#stack = np.asarray(stack)

examples_dir = 'Y:/lightsheet stuff/20210406 Deskew of dot pattern_argolight/'

#if not os.path.exists(examples_dir):
    #os.mkdir(examples_dir)
#fname = examples_dir + "ex1_t2_ch2_illum2_angle2.h5"
from PIL import Image
image_name = examples_dir + "ch0_y0/Argolight_field_of_dots_NDTiffStack_crop.tif"
image_plane = Image.open(image_name )
image_plane = np.array(image_plane)
stack_dim_zyx = (1,512,336)

fname = examples_dir + "ex8_virtual_stack_missingsssssssss.h5"
print("xxxxxxxxxxxx")
bdv_writer = npy2bdv.BdvWriter(fname, nchannels=1, subsamp=((1, 1, 1),))
print("xxxxxxxxxxxx")
#bdv_writer.append_view(stack=None, virtual_stack_dim=stack_dim_zyx, time=0, channel=0)
#bdv_writer.append_view(stack=None, virtual_stack_dim=stack_dim_zyx, time=0, channel=1)
bdv_writer.append_view(stack=None, virtual_stack_dim=(4,512,336))
for i_plane in range(stack_dim_zyx[0]):
    print(i_plane)
    bdv_writer.append_plane(plane=image_plane, plane_index=i_plane, channel=0)