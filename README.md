# Key-Point-Detection-Key-Point-Matching-and-Image-Stitching


Implemented key point detection using SIFT (Scale-Invariant Feature Transform) from scikit-image, a popular computer vision library.

Developed a key point matching function that takes two sets of key point features and returns a list of indices of matching pairs, establishing correspondences between two images.

Created a visualization function that combines two images side-by-side and plots the detected key points and their matched connections, providing a visual representation of key point correspondences.

Implemented an image stitching solution that computes a transformation matrix used to warp one image for stitching with another image, utilizing key points matched between the two images.

Developed functions to compute affine and projective transformation matrices based on matched key point coordinates, using the normal equations approach. The functions return 2x3 and 3x3 matrices, respectively.

Implemented the RANSAC (Random Sample Consensus) algorithm for robust estimation of the transformation matrix by identifying and rejecting outlier matches, improving the accuracy and reliability of the image stitching process.
