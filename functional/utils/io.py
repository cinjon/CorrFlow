import imageio
from PIL import Image
import numpy as np

import os

PWD = os.path.dirname(os.path.realpath(__file__))
PATH_PALETTE = PWD+'/../../datas/palette.txt'
default_palette = np.loadtxt(PATH_PALETTE,dtype=np.uint8).reshape(-1,3)

def imread_indexed(filename):
  """ Load image given filename."""

  im = Image.open(filename)

  annotation = np.atleast_3d(im)[...,0]
  ret = np.array(im.getpalette()).reshape((-1,3))
  # /checkpoint/cinjon/spaceofmotion/supercons/davis/DAVIS/Annotations/480p/gold-fish/00000.png (480, 854) (256, 3)
  # print('imread: ', filename, annotation.shape, ret.shape)
  return annotation, ret

def imwrite_indexed(filename,array,color_palette=default_palette):
  """ Save indexed png."""

  if np.atleast_3d(array).shape[2] != 1:
    raise Exception("Saving indexed PNGs requires 2D array.")

  im = Image.fromarray(array)
  im.putpalette(color_palette.ravel())
  im.save(filename, format='PNG')

def write_img(filename, array):
  imageio.imwrite(filename, array)
  # im = Image.fromarray(array)
  # im.save(filename, format='PNG')
  
