

import numpy as np
from PIL import Image

import os
import glob
import subprocess

path = 'results/w7x_ar16_op12b_slab_1500eV'

filename_list = glob.glob(os.path.join(path, 'xicsrt_detector*.tif'))

for ff in filename_list:
    print(ff)


image = Image.open(filename_list[0])
image_array = np.array(image)

for ff in filename_list[1:]:
    image = Image.open(ff)
    image_array += np.array(image)    

# Add some noise.
# This is needed to provide proper statistics during spectal fitting.
# The idea is to add just enough noise that the Gaussian distribution
# starts to approximate the Poisson distribution, but also keep the
# signal to noise ratio down in tha case of small signal.
#noise_level = min(10, np.max(image_array)*0.05)
noise_level = 10
noise = np.random.poisson(noise_level, image_array.shape)
image_array += noise

# This make the image match the output from the w7x_ar16 pilatus detector.
# This is correct for the Ar16+ system.
# Both a flip and a transpose are required.
out = Image.fromarray(np.flip(image_array,axis=0).T)
out.save('temp_detector_w7x_ar16.tif')

# Now move the image into the standard shot format.
subprocess.run('mkdir /u/npablant/data/xics/w7x/w7x_ar16/189000000/images',shell=True)
subprocess.run('cp temp_detector_w7x_ar16.tif /u/npablant/data/xics/w7x/w7x_ar16/189000000/images/w7x_ar16_189000000_00000.tif',shell=True)
subprocess.run('zip -rj /u/npablant/data/xics/w7x/w7x_ar16/189000000/w7x_ar16_189000000.zip /u/npablant/data/xics/w7x/w7x_ar16/189000000/images', shell=True)
