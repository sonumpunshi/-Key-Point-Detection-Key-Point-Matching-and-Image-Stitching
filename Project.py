import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

img1 = cv2.imread('campus_002.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('campus_003.jpg', cv2.IMREAD_COLOR)

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(gray1, None)
kp2, desc2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors
matches = bf.match(desc1, desc2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matching keypoints
src_pts = [kp1[m.queryIdx].pt for m in matches]
dst_pts = [kp2[m.trainIdx].pt for m in matches]


def plot_matches(img1, img2, kp1, kp2, matches):
    # Draw the matches on the images
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)

    # Display the images
    plt.imshow(match_img)
    plt.show()


plt.imshow(cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0)))
plt.show()

plt.imshow(cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0)))
plt.show()

plot_matches(img1, img2, kp1, kp2, matches)


def compute_affine_transform(src_pts, dst_pts):
    # Convert the point lists to numpy arrays
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Compute the affine transformation matrix
    A = np.zeros((2 * len(src_pts), 6))
    b = np.zeros((2 * len(src_pts), 1))

    for i in range(len(src_pts)):
        A[2 * i, :] = [src_pts[i][0], src_pts[i][1], 0, 0, 1, 0]
        A[2 * i + 1, :] = [0, 0, src_pts[i][0], src_pts[i][1], 0, 1]
        b[2 * i, 0] = dst_pts[i][0]
        b[2 * i + 1, 0] = dst_pts[i][1]

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    M = np.zeros((3, 3))
    M[0, 0] = x[0]
    M[0, 1] = x[1]
    M[0, 2] = x[4]
    M[1, 0] = x[2]
    M[1, 1] = x[3]
    M[1, 2] = x[5]
    M[2, 0] = 0
    M[2, 1] = 0
    M[2, 2] = 1

    return M


def compute_projective_transform(src_pts, dst_pts):
    # Convert the point lists to numpy arrays
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Compute the projective transformation matrix
    A = np.zeros((2 * len(src_pts), 8))
    b = np.zeros((2 * len(src_pts), 1))

    for i in range(len(src_pts)):
        A[2 * i, :] = [src_pts[i][0], src_pts[i][1], 1, 0, 0, 0, -src_pts[i][0] * dst_pts[i][0],
                       -src_pts[i][1] * dst_pts[i][0]]
        A[2 * i + 1, :] = [0, 0, 0, src_pts[i][0], src_pts[i][1], 1, -src_pts[i][0] * dst_pts[i][1],
                           -src_pts[i][1] * dst_pts[i][1]]
        b[2 * i, 0] = dst_pts[i][0]
        b[2 * i + 1, 0] = dst_pts[i][1]

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    M = np.zeros((3, 3))
    M[0, 0] = x[0]
    M[0, 1] = x[1]
    M[0, 2] = x[2]
    M[1, 0] = x[3]
    M[1, 1] = x[4]
    M[1, 2] = x[5]
    M[2, 0] = x[6]
    M[2, 1] = x[7]
    M[2, 2] = 1

    return M


def ransac(src_pts, dst_pts, num_iterations, sample_size, threshold):
    best_model = None
    best_num_inliers = 0

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    for i in range(num_iterations):
        # Sample a subset of points
        sample_indices = random.sample(range(len(src_pts)), sample_size)
        src_sample = src_pts[sample_indices]
        dst_sample = dst_pts[sample_indices]

        # Compute the transformation matrix
        M = compute_projective_transform(src_sample, dst_sample)
        # M = compute_affine_transform(src_sample, dst_sample)
        # Apply the transformation to all points
        src_pts_homogeneous = np.hstack((src_pts, np.ones((len(src_pts), 1))))
        src_pts_transformed_homogeneous = np.dot(M, src_pts_homogeneous.T).T
        src_pts_transformed = src_pts_transformed_homogeneous[:, :2] / src_pts_transformed_homogeneous[:, 2:]

        # Compute the number of inliers
        dists = np.linalg.norm(src_pts_transformed - dst_pts, axis=1)
        num_inliers = np.sum(dists < threshold)

        # Check if this is the best model so far
        if num_inliers > best_num_inliers:
            best_model = M
            best_num_inliers = num_inliers

    return best_model




def warp_image(image, transform_matrix):
    rows, cols, _ = image.shape
    warped_image = cv2.warpPerspective(image, transform_matrix, (cols, rows))
    y_offset, x_offset = np.abs(transform_matrix[:2, 2]).astype(np.int)
    warped_image = cv2.copyMakeBorder(warped_image, y_offset, 0, x_offset, 0, cv2.BORDER_CONSTANT, value=0)
    return warped_image


def stitch_images(img1, img2, M):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Find the corners of the second image and transform them to the first image space
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_warped = cv2.perspectiveTransform(pts2, M)

    # Find the maximum x and y coordinates to determine the output size
    max_x = max(h1, pts2_warped[:, :, 0].max())
    max_y = max(w1, pts2_warped[:, :, 1].max())
    dst_size = (int(max_y), int(max_x))

    # Warp the second image to the first image space
    img2_warped = cv2.warpPerspective(img2, M, dst_size)

    # Resize the warped image to match the size of the first image
    img2_warped = cv2.resize(img2_warped, (w1, h1))

    # Merge the two images by taking the maximum pixel values
    stitched_image = np.maximum(img1, img2_warped)

    return stitched_image


M = ransac(src_pts, dst_pts, num_iterations=500, sample_size=4, threshold=10)

# Warp the source image using the transformation matrix
stitched = stitch_images(img1, img2, M)
stitched_image_normalized = cv2.normalize(stitched, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('stitched.jpg', stitched_image_normalized)
imgnew = plt.imread('stitched.jpg')
plt.title('Yosemite 1 and Yosemite Projective')
plt.imshow(imgnew)
plt.show()
