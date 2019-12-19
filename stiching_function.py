import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
from PIL import Image

def compute_homography_naive(mp_src, mp_dst):
    '''
        Input:
        A variable containing 2 rows and N columns, where the i column
        represents coordinates of match point i in the src image.
        mp_src –
        A variable containing 2 rows and N columns, where the i column
        represents coordinates of match point i in the dst image.
        mp_dst –
        Output :
        H – Projective transformation matrix from src to dst
        '''

    mp_src = np.vstack((mp_src, np.ones((1, mp_src.shape[1]))))  # adding 3rd coordinate equals always to 1
    mp_src = mp_src.T
    mp_dst = mp_dst.T
    zeros_mat = np.zeros_like(mp_src)  # zeros sized n x 3
    dst_x_part = np.hstack((mp_src, zeros_mat, -(mp_dst[:, 0] * mp_src.T).T))
    dst_y_part = np.hstack((zeros_mat, mp_src, -(mp_dst[:, 1] * mp_src.T).T))
    A = np.vstack((dst_x_part, dst_y_part))

    e_vals, e_vecs = np.linalg.eig(A.T @ A)
    h = e_vecs[:, e_vals.argmin()]
    h /= h[-1]
    H = h.reshape((3, 3))
    return H

def compute_homography(mp_src, mp_dst, inliers_percent, max_err):
    # RANSAC number of iterations: np.log(1-p)/np.log(1-w^n)
    # here w is the inliers_perscent and minimal number of points to solve the model is 4.
    # assuming success probability of 99.99% and inliers percent of 50% we get:
    # np.log(1 - 0.9999) / np.log(1 - 0.5 ** 4)  ~= 140 iterations.
    # so assuming inliers_percent > 50%, we can set p=0.9999 and get a reasonable amount of iterations.
    from random import sample
    N = mp_src.shape[1] # number of matching points.
    n = 4 # minimum number of required points to solve model.
    K = int(np.ceil(np.log(1-0.9999)/np.log(1-inliers_percent**n))) # number of RANSAC iterations.
    min_err = np.inf # initial the minimal error with inf value.

    for k in range(K*5):
        # 'sample' guarantees unique random integers / no repetitions
        idx = sample(range(N), n)
        H = compute_homography_naive(mp_src[:, idx], mp_dst[:, idx])
        fit_percent, dist_mse = test_homography(H, mp_src, mp_dst, max_err)
        if fit_percent < inliers_percent: continue
        # set 1 to recalculate H with all inliers. set 0 to skip this step.
        if 1:
            H = compute_homography_inliers(H, mp_src, mp_dst, max_err)
            fit_percent, dist_mse = test_homography(H, mp_src, mp_dst, max_err)
        if dist_mse < min_err:
            min_err = dist_mse
            H_ransac = H
    # print(min_err)
    return H_ransac

def compute_homography_inliers(H, mp_src, mp_dst, max_err):
    mp_src = np.vstack((mp_src, np.ones((1, mp_src.shape[1]))))  # adding 3rd coordinate equals always to 1
    mp_dst_hat = H@mp_src
    # divide by 3rd coordinate
    mp_dst_hat /= mp_dst_hat[-1, :]
    diff_vec = np.abs(mp_dst_hat[:-1, :]-mp_dst)
    diff_vec = np.linalg.norm(diff_vec, axis=0)
    inliers_mask = diff_vec <= max_err
    mp_src = mp_src[:-1, :]
    H = compute_homography_naive(mp_src[:, inliers_mask], mp_dst[:, inliers_mask])
    return H

def test_homography(H, mp_src, mp_dst, max_err):
    mp_src = np.vstack((mp_src, np.ones((1, mp_src.shape[1]))))  # adding 3rd coordinate equals always to 1
    mp_dst_hat = H@mp_src
    # divide by 3rd coordinate
    mp_dst_hat /= mp_dst_hat[-1, :]
    diff_vec = np.abs(mp_dst_hat[:-1, :]-mp_dst)
    diff_vec = np.linalg.norm(diff_vec, axis=0)
    inliers_mask = diff_vec <= max_err
    fit_percent = inliers_mask.sum()/inliers_mask.size
    dist_mse = (diff_vec[inliers_mask]**2).mean()

    # return fit_percent, dist_mse,inliers_mask
    return fit_percent, dist_mse

def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err):
    # find projected image:
    H = compute_homography(mp_src, mp_dst, inliers_percent, max_err)
    prj_img = backward_mapping(H, img_src)

    # find offset:
    src_x = img_src.shape[1]
    src_y = img_src.shape[0]
    p = get_corners(H, src_x - 1, src_y - 1)
    offset_x, offset_y = np.round(p.min(0)).astype('int')

    dim_x = np.max((prj_img.shape[1], img_dst.shape[1]-offset_x))
    dim_y = np.max((prj_img.shape[0], img_dst.shape[0]-offset_y))

    img_pan = np.zeros((dim_y, dim_x, 3), dtype='int')
    img_pan_temp = np.zeros((dim_y, dim_x, 3), dtype='int')
    img_pan_temp[-offset_y:(-offset_y+img_dst.shape[0]), -offset_x:(-offset_x+img_dst.shape[1])] = img_dst
    img_pan[:prj_img.shape[0], :prj_img.shape[1]] = prj_img
    mask = img_pan_temp > 0
    img_pan[mask] = img_pan_temp[mask]
    return img_pan

def get_corners(H, src_x, src_y):
    p = H@np.array([[0, 0, 1], [0, src_y, 1], [src_x, 0, 1], [src_x, src_y, 1]]).T
    p /= p[-1, :]
    p = p[:-1, :].T
    return p

def forward_mapping(H, img_src):
    # define dst grid:
    src_x = img_src.shape[1]
    src_y = img_src.shape[0]

    # project src points to dst grid:
    xx, yy = np.meshgrid(range(src_x), range(src_y))
    src_coords = np.vstack((xx.flatten(), yy.flatten(), np.ones_like(yy.flatten())))

    src_coords = H@src_coords
    src_coords /= src_coords[-1, :]
    src_coords = src_coords[:-1, :]
    src_coords = np.round(src_coords).astype('int').T
    src_coords -= src_coords.min(0)

    # apply the forward mapping:
    nc, nr = src_coords.max(0) + 1
    # handle bad projections:
    if np.any(np.array([nc, nr]) > 5000):
        src_coords = src_coords / src_coords.std(0).max()
        src_coords = src_coords - src_coords.mean(0)
        upper_limit = src_coords.mean(0)+1*src_coords.std(0)
        lower_limit = src_coords.mean(0)-1*src_coords.std(0)
        idx = (src_coords[:, 0] <= upper_limit[0]) * (src_coords[:, 1] <= upper_limit[1])*(src_coords[:, 0] >= lower_limit[0]) * (src_coords[:, 1] >= lower_limit[1])
        src_coords = src_coords[idx, :]
        src_coords -= src_coords.min(0)
        src_coords *= np.sqrt(src_coords.shape[0]/src_coords.max(0).prod())
        src_coords = np.round(src_coords).astype('int')
        nc, nr = src_coords.max(0) + 1
        yy = yy.flatten()[idx]
        xx = xx.flatten()[idx]

    dst_img = np.zeros((nr, nc, 3), dtype='int')
    dst_img[src_coords[:, 1], src_coords[:, 0], :] = img_src[yy.flatten(), xx.flatten(), :]
    return dst_img

def backward_mapping(H, img_src):
    # define dst grid:
    src_x = img_src.shape[1]
    src_y = img_src.shape[0]
    p = get_corners(H, src_x - 1, src_y - 1)
    p_min = np.round(p.min(0)).astype('int')
    p_max = np.round(p.max(0)).astype('int')
    xx, yy = np.meshgrid(range(p_min[0], p_max[0]), range(p_min[1], p_max[1]))
    dst_coords = np.vstack((xx.flatten(), yy.flatten(), np.ones_like(yy.flatten())))

    # extract src points of dst grid:
    H_inv = np.linalg.inv(H)
    src_coords = H_inv@dst_coords
    src_coords /= src_coords[-1, :]
    src_coords = src_coords[:-1, :]

    # mark all out of bounds coordinates:
    idx = (src_coords[0, :] >= 0)*(src_coords[0, :] <= (src_x-1))*(src_coords[1, :] >= 0)*(src_coords[1, :] <= (src_y-1))
    xy = src_coords[:, idx]

    # interpolate on the backward estimated src_coords:
    Q11 = np.floor(xy)
    Q22 = np.ceil(xy)
    Q12 = np.vstack((Q11[0, :], Q22[1, :]))
    Q21 = np.vstack((Q22[0, :], Q11[1, :]))
    dst_img = np.zeros_like(dst_coords)
    dst_img[:, idx] = (img_src[Q11[1, :].astype('int'), Q11[0, :].astype('int')].T*(Q22-xy).prod(axis=0)
               + img_src[Q22[1, :].astype('int'), Q22[0, :].astype('int')].T*(xy-Q11).prod(axis=0)
               - img_src[Q21[1, :].astype('int'), Q21[0, :].astype('int')].T*(xy-Q12).prod(axis=0)
               - img_src[Q12[1, :].astype('int'), Q12[0, :].astype('int')].T*(xy-Q21).prod(axis=0)) / (Q22-Q11).prod(axis=0)
    dst_img = dst_img.T.reshape(xx.shape[0], xx.shape[1], -1)
    return dst_img

# debug scripts:
def plotImgsAndPoints(img_src, img_dst, matchesPoints):
    mp_dst = matchesPoints['match_p_dst'].astype(float)
    mp_src = matchesPoints['match_p_src'].astype(float)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img_src)
    axarr[0].scatter(mp_src[0], mp_src[1], c='r', s=1)
    axarr[1].imshow(img_dst)
    axarr[1].scatter(mp_dst[0], mp_dst[1], c='r', s=1)

def runPart_A(img_src,img_dst):
    # compute naive homography:
    print("\n\n==============\n Showing results for naive homography:\n==============\n")
    matches = scipy.io.loadmat('matches')
    mp_dst = matches['match_p_dst'].astype(float)
    mp_src = matches['match_p_src'].astype(float)

    H = compute_homography_naive(mp_src, mp_dst)
    print("homography results: \n")
    print(H)
    print('==============================')

    newimg = forward_mapping(H, img_src)
    plt.imshow(newimg)
    plt.title('homography with Forward mapping')
    plt.show()


def runPart_B(img_src,img_dst):
    # compute ransac homography:
    print("\n\n==============\n Showing results for RANSAC homography:\n==============\n")
    matches = scipy.io.loadmat('matches')
    mp_dst = matches['match_p_dst'].astype(float)
    mp_src = matches['match_p_src'].astype(float)

    H = compute_homography(mp_src, mp_dst, 0.8, 15)
    print("RANSAC homography results: \n")
    print(H)
    print('==============================')

    newimg = forward_mapping(H, img_src)
    plt.imshow(newimg)
    plt.title('RANSAC homography with Forward mapping')
    plt.show()

def runPart_C(img_src,img_dst):
    # compute ransac homography:
    print("\n\n==============\n Showing results for RANSAC homography:\n==============\n")
    matches = scipy.io.loadmat('matches')
    mp_dst = matches['match_p_dst'].astype(float)
    mp_src = matches['match_p_src'].astype(float)

    H = compute_homography(mp_src, mp_dst, 0.8, 15)
    print("RANSAC homography results: \n")
    print(H)
    print('==============================')

    newimg = backward_mapping(H, img_src)
    plt.imshow(newimg)
    plt.title('RANSAC homography with Backward mapping')
    plt.show()

def runPanorma(img_src,img_dst):
    matches = scipy.io.loadmat('matches')
    mp_dst = matches['match_p_dst'].astype(float)
    mp_src = matches['match_p_src'].astype(float)

    inliers_percent = 0.8
    max_err = 10

    img_pan = panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err)
    plt.imshow(img_pan)
    plt.title('Stiching the src image to dst')
    plt.show()


# Test the functions
if __name__ == '__main__':

    # load the images and show the matching points:
    img_src = mpimg.imread('src.jpg')
    img_dst = mpimg.imread('dst.jpg')
    
    # naive, all matching points homography + forward mapping.
    runPart_A(img_src, img_dst)
    # ransac homography + forward mapping.
    runPart_B(img_src, img_dst)
    # ransac homography + backward mapping.
    runPart_C(img_src, img_dst)

    runPanorma(img_src, img_dst)

    print("\n\n===================\n\tScript Done\n\n")
