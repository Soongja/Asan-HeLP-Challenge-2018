import os
import numpy as np
import SimpleITK as sitk

image = sitk.ReadImage('sample_cardiac/CHD/image/sample_dia.mha')
label = sitk.ReadImage('sample_cardiac/CHD/label/sample_dia_M.mha')
label = sitk.LabelToRGB(label)

for im in [image, label]:
    pixel_type = im.GetPixelID()
    size = im.GetSize()
    spacing = im.GetSpacing()
    origin = im.GetOrigin()
    direction = im.GetDirection()

    print('pixel type:', pixel_type, 'size:', size, 'spacing:', spacing, 'origin:', origin, 'direction:', direction)

image = sitk.GetArrayFromImage(image)
label = sitk.GetArrayFromImage(label)
print(image.dtype)
print(label.dtype)

image = sitk.GetImageFromArray(image)
label = sitk.GetImageFromArray(label)

for im in [image, label]:
    pixel_type = im.GetPixelID()
    size = im.GetSize()
    spacing = im.GetSpacing()
    origin = im.GetOrigin()
    direction = im.GetDirection()

    print('pixel type:', pixel_type, 'size:', size, 'spacing:', spacing, 'origin:', origin, 'direction:', direction)