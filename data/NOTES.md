This file is placeholder for the dataset.

We are not publicly uploading this dataset but you can place it here under
directory named `raw` with subdirectories: `Train`, `Test`, `Validate`.

A helper script will convert the images into python numpy binarized files for
easier use.

This script will pad images to the smallest common size (maxima on X and Y) with
a black regions. If this disrupts the algorithm then try doing facial detection
and cropping a smaller patch first.
