import h5py
import numpy as np
from PIL import Image
import npy2bdv

#hdf = h5py.File("Y:/lightsheet stuff/20210406 Deskew of dot pattern_argolight/deskewed_ch0.h5",'r')
file = "Y:/lightsheet stuff/20210406 Deskew of dot pattern_argolight/deskewed_ch0.h5"
#file = "C:/Users/Peach/anaconda3/Lib/site-packages/npy2bdv/examples/ex1_t2_ch2_illum2_angle2.h5"
bdv_reader = npy2bdv.BdvReader(file)
stack = bdv_reader.read_view(time=0, isetup=0, ilevel=0)
bdv_reader.close()