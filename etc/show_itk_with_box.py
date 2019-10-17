import os
import cv2
import numpy as np
import SimpleITK as sitk

# image = sitk.ReadImage('sample_cardiac/CHD/image/sample_sys.mha')
image = sitk.ReadImage('sample_cardiac/HCMP/image/sample.mha')
label = sitk.ReadImage('sample_cardiac/HCMP/label/sample_M.mha')
# image = sitk.ReadImage('sample_dia_output.mha')

voxel_space = image.GetSpacing()
origin = image.GetOrigin()
print(voxel_space)
print(origin)

# image = sitk.GetArrayFromImage(image) * 80
image = sitk.GetArrayFromImage(image)
label = sitk.GetArrayFromImage(label) * 80

max_val = np.max(image)
min_val = np.min(image)

image = (image - min_val) / (max_val - min_val)
image = (image * 255).astype(np.uint8)

for i in range(image.shape[0]):
    cv2.putText(image[i], '#%s: %s x %s x %s' % (i, image.shape[0], image.shape[1], image.shape[2]),
                (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, [255], 1)

    center = 128
    boxL = 200
    box = np.array([[center - boxL / 2, center - boxL / 2], [center + boxL / 2, center - boxL / 2],
                    [center + boxL / 2, center + boxL / 2], [center - boxL / 2, center + boxL / 2]], dtype=np.int32)
    cv2.drawContours(image[i], [box], -1, 255, 2)
    cv2.drawContours(label[i], [box], -1, 255, 2)

    cv2.imshow('image', image[i])
    cv2.imshow('label', label[i])

    k = cv2.waitKey(0)
    if k == 27:
        break

cv2.destroyAllWindows()
