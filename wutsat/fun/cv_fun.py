from wutsat.fun import config as cf


def detect(im1, im2, keypoints_method='sift', matches_method='flann', ransac_th=5.0, MIN_MATCH_COUNT=10,
           lowe_ratio=0.75):
    """TODO: Description to be provided - detects keypoint then finds matches using given method with given parameters
Parameters
----------
im1 : string
    Path to first image
im2 : string
    Path to second image
method : string
    Name of method used. Supported methods:
Returns
-------
new_lats: list of floats
    List of latitudes.
"""
    # TODO: Add method verification so it will throw an error when incorrect is given
    import numpy as np
    import cv2 as cv
    error = [0, '']

    # Read images suports color
    img1 = cv.imread(im1, 1)
    img2 = cv.imread(im2, 1)

    # Choose method of finding keypoints
    if keypoints_method == 'sift':
        meth = cv.xfeatures2d.SIFT_create()
    elif keypoints_method == 'surf':
        meth = cv.xfeatures2d.SURF_create()
    elif keypoints_method == 'orb':
        meth = cv.ORB_create()

    # Exporting to grayscale
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Detect and compute keypoints
    kp1, des1 = meth.detectAndCompute(gray1, None)
    kp2, des2 = meth.detectAndCompute(gray2, None)

    # Check if keypoints exists if not return error
    if (kp1 is None or kp2 is None or des1 is None or des2 is None):
        a, b, c, d = type(kp1), type(kp2), type(des1), type(des2)
        error = [1, 'No keypoints or descriptors']
        # TODO: not sure what we will return
        return (error, None, None)

    if (matches_method == 'flann'):
        if (keypoints_method == 'orb'):
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        # Number of times the trees in the index should be recursively traversed, greter = better
        search_params = dict(checks=100)
        matches_m = cv.FlannBasedMatcher(index_params, search_params)
    elif matches_method == 'bf':
        matches_m = cv.BFMatcher()

    # Check if descriptor length is valid
    if len(des1) < 2 or len(des2) < 2:
        error = [2, 'Descriptors length invalid']
        return (error, None, None)

    matches = matches_m.knnMatch(des1, des2, k=2)

    # Collect all good matches
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        (m, n) = m_n
        if m.distance < lowe_ratio * n.distance:
            good.append(m)
    # for m,n in matches:
    #     if m.distance < lowe_ratio*n.distance:
    #         good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_th)
        matchesMask = None
        if M is not None and mask is not None:
            matchesMask = mask.ravel().tolist()

            h, w, d = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            dst = cv.perspectiveTransform(pts, M)
            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    if 'dst' not in locals():
        dst = None

    imgs = [img1.shape, img2.shape, im1, im2]
    return (error, img3, dst, imgs)


def find_corresponding_photos_from_file(save_path):
    from wutsat.fun import mat_fun
    methods = mat_fun.rwdata(save_path, 'all_methods.pkl', 'r', None)
    results = mat_fun.check_all_methods(methods)

    if len(results) > 2:
        raise Exception('results expects only one value, more are given: {}'.format(results))
    if not results[0]:
        raise Exception('List is empty, nothing was found :(')
    extent = methods[results[0][0]][results[1][0]]
    return extent, methods, results


def find_corresponding_photos(directory, img, result_path, save_photos=0,
                              save_path=cf.fcp_methods, calculate_only=False, read_only=False):
    from wutsat.fun import mat_fun
    import os
    sorted_files = sorted(os.scandir(directory), key=lambda t: t.stat().st_mtime)
    files = [f.name for f in sorted_files]
    # TODO: smarter way to get list(tulpe) of images, it would be better to have them passed not read from directory
    # TODO: Get rid of 'data_set'
    data_set = None
    photos = [directory + '/' + str(i) + '.png' for i in range(len(files))]
    if os.path.isfile(save_path + '/all_methods.pkl'):
        read_saved = True
        save = False
    else:
        read_saved = False
        save = True

    if read_saved:
        methods = mat_fun.rwdata(save_path, 'all_methods.pkl', 'r', None)
    else:
        # print('ans1 starting')
        ans = check_meth4('sift', 'bf', result_path, photos, img, files, data_set, save_photos)
        ans2 = check_meth4('sift', 'flann', result_path, photos, img, files, data_set, save_photos)
        ans3 = check_meth4('surf', 'flann', result_path, photos, img, files, data_set, save_photos)
        ans4 = check_meth4('surf', 'bf', result_path, photos, img, files, data_set, save_photos)
        ans5 = check_meth4('orb', 'bf', result_path, photos, img, files, data_set, save_photos)
        ans6 = check_meth4('orb', 'flann', result_path, photos, img, files, data_set, save_photos)
        methods = [ans, ans2, ans3, ans4, ans5, ans6]
        # methods = [ans]
        # print('all methods done')
        if save:
            mat_fun.rwdata(save_path, 'all_methods.pkl', 'w', methods)
    if calculate_only:
        return

    results = mat_fun.check_all_methods(methods)
    if len(results) > 2:
        raise Exception('results expects only one value, more are given: {}'.format(results))
    if not results[0]:
        raise Exception('List is empty, nothing was found :(')
    extent = methods[results[0][0]][results[1][0]]
    return extent, methods, results
