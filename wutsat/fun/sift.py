def sift_flan(im1,im2,lowe_ratio,min_match,display,ransac_th):
	import numpy as np
	import cv2 as cv
	from matplotlib import pyplot as plt

	#Reading images
	img1 = cv.imread(im1,1)
	img2 = cv.imread(im2,1)
	#Making sift object - it takes default but also can take
	#(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma)
	sift = cv.xfeatures2d.SIFT_create()
	#Exporting to grayscale
	gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
	gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	#Detecting keypoints and computing descriptors for every keypoint
	kp1, des1 = sift.detectAndCompute(gray1,None)
	kp2, des2 = sift.detectAndCompute(gray2,None)

	if (kp1 is None or kp2 is None or des1 is None or des2 is None):
		a,b,c,d = type(kp1), type(kp2), type(des1), type(des2)
		return(a,b,c,d,None)

	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	#Number of times the trees in the index should be recursively traversed, greter = better
	search_params = dict(checks = 100)
	flann = cv.FlannBasedMatcher(index_params, search_params)
	if len(des1) < 2 or len(des2) < 2:
		return(None, None, None, None, None)
	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	if lowe_ratio == None:
		lowe_ratio = 0.75

	good = []
	for m,n in matches:
		if m.distance < lowe_ratio*n.distance:
			good.append(m)

	if min_match == None:
		MIN_MATCH_COUNT = 10
	else:
		MIN_MATCH_COUNT = min_match

	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		if ransac_th== None:
			ransac_th = 5.0

		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,ransac_th)
		matchesMask = None
		if M is not None and mask is not None:
			matchesMask = mask.ravel().tolist()

			h,w,d = img1.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

			dst = cv.perspectiveTransform(pts,M)
			img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)
	img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	if display != None:
			plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB)),plt.axis("off"),plt.show()
	return(img1,img2,img3,good,draw_params)


def sift_bfma(im1,im2,lowe_ratio,min_match,display,ransac_th):
	import numpy as np
	import cv2 as cv
	from matplotlib import pyplot as plt

	#Reading images
	img1 = cv.imread(im1,1)
	img2 = cv.imread(im2,1)
	#Making sift object - it takes default but also can take
	#(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma)
	sift = cv.xfeatures2d.SIFT_create()
	#Exporting to grayscale
	gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
	gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	#Detecting keypoints and computing descriptors for every keypoint
	kp1, des1 = sift.detectAndCompute(gray1,None)
	kp2, des2 = sift.detectAndCompute(gray2,None)

	if (kp1 is None or kp2 is None or des1 is None or des2 is None):
		a,b,c,d = type(kp1), type(kp2), type(des1), type(des2)
		return(a,b,c,d,None)

	bf = cv.BFMatcher()
	if len(des1) < 2 or len(des2) < 2:
		return(None,None,None,None,None)
	matches = bf.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	if lowe_ratio == None:
		lowe_ratio = 0.75

	good = []
	for m,n in matches:
		if m.distance < lowe_ratio*n.distance:
			good.append(m)

	if min_match == None:
		MIN_MATCH_COUNT = 10
	else:
		MIN_MATCH_COUNT = min_match

	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		if ransac_th== None:
			ransac_th = 5.0

		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,ransac_th)
		matchesMask = None
		if M is not None and mask is not None:
			matchesMask = mask.ravel().tolist()

			h,w,d = img1.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv.perspectiveTransform(pts,M)

			img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)
	img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	if display != None:
			plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB)),plt.axis("off"),plt.show()
	return(img1,img2,img3,good,draw_params)

def surf_flan(im1,im2,lowe_ratio,min_match,display,ransac_th):
	import numpy as np
	import cv2 as cv
	from matplotlib import pyplot as plt

	#Read images
	img1 = cv.imread(im1,1)
	img2 = cv.imread(im2,1)
	#Making sift object - it takes default but also can take
	#(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma)
	surf = cv.xfeatures2d.SURF_create()
	#Exporting to grayscale
	gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
	gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	#Detecting keypoints and computing descriptors for every keypoint
	kp1, des1 = surf.detectAndCompute(gray1,None)
	kp2, des2 = surf.detectAndCompute(gray2,None)

	if (kp1 is None or kp2 is None or des1 is None or des2 is None):
		a,b,c,d = type(kp1), type(kp2), type(des1), type(des2)
		return(a,b,c,d,None)
	if len(kp1) < 2 or len(kp2) < 2:
		return(None,None,None,None,None)

	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

	search_params = dict(checks = 100)
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	#Store all the good matches as per Lowe's ratio test.
	if lowe_ratio == None:
		lowe_ratio = 0.75

	good = []
	for m,n in matches:
		if m.distance < lowe_ratio*n.distance:
			good.append(m)

	if min_match == None:
		MIN_MATCH_COUNT = 10
	else:
		MIN_MATCH_COUNT = min_match

	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		if ransac_th== None:
			ransac_th = 5.0

		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,ransac_th)
		matchesMask = None

		if M is not None and mask is not None:
			matchesMask = mask.ravel().tolist()
			h,w,d = img1.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv.perspectiveTransform(pts,M)

			img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)
	img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	if display != None:
			plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB)),plt.axis("off"),plt.show()
	return(img1,img2,img3,good,draw_params)

def surf_bfma(im1,im2,lowe_ratio,min_match,display,ransac_th):
	import numpy as np
	import cv2 as cv
	from matplotlib import pyplot as plt
	from PIL import Image

	#Read images
	img1 = cv.imread(im1,1)
	img2 = cv.imread(im2,1)
	#Making sift object - it takes default but also can take
	#(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma)
	surf = cv.xfeatures2d.SURF_create()
	#Exporting to grayscale
	gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
	gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	#Detecting keypoints and computing descriptors for every keypoint
	kp1, des1 = surf.detectAndCompute(gray1,None)
	kp2, des2 = surf.detectAndCompute(gray2,None)

	if (kp1 is None or kp2 is None or des1 is None or des2 is None):
		a,b,c,d = type(kp1), type(kp2), type(des1), type(des2)
		return(a,b,c,d,None)
	if len(kp1) < 2 or len(kp2) < 2:
		return(None,None,None,None,None)

	bf = cv.BFMatcher()

	matches = bf.knnMatch(des1,des2,k=2)
	#Store all the good matches as per Lowe's ratio test.
	if lowe_ratio == None:
		lowe_ratio = 0.75

	good = []
	for m,n in matches:
		if m.distance < lowe_ratio*n.distance:
			good.append(m)

	if min_match == None:
		MIN_MATCH_COUNT = 10
	else:
		MIN_MATCH_COUNT = min_match

	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		if ransac_th== None:
			ransac_th = 5.0

		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,ransac_th)
		matchesMask = None

		if M is not None and mask is not None:
			matchesMask = mask.ravel().tolist()
			h,w,d = img1.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv.perspectiveTransform(pts,M)

			img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)
	img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	if display != None:
			plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB)),plt.axis("off"),plt.show()
	return(img1,img2,img3,good, draw_params)

def orb_bfma(im1,im2,f_num,min_match,display,ransac_th):
	import numpy as np
	import cv2 as cv
	from matplotlib import pyplot as plt
	from PIL import Image

	#Read images
	img1 = cv.imread(im1,1)
	img2 = cv.imread(im2,1)
	#Making orb object - it takes default but also can take
	if f_num is None:
		f_num = 16000
	orb = cv.ORB_create(f_num)
	#Exporting to grayscale
	gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
	gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	#Detecting keypoints and computing descriptors for every keypoint
	kp1, des1 = orb.detectAndCompute(gray1,None)
	kp2, des2 = orb.detectAndCompute(gray2,None)

	if (kp1 is None or kp2 is None or des1 is None or des2 is None):
		a,b,c,d = type(kp1), type(kp2), type(des1), type(des2)
		return(a,b,c,d,None)

	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)

	if min_match == None:
		MIN_MATCH_COUNT = 10
	else:
		MIN_MATCH_COUNT = min_match

	if len(matches)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

		if ransac_th== None:
			ransac_th = 5.0

		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,ransac_th)
		matchesMask = mask.ravel().tolist()

		h,w,d = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv.perspectiveTransform(pts,M)

		img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT) )
		matchesMask = None

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)
	img3 = cv.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
	if display != None:
			plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB)),plt.axis("off"),plt.show()
	return(img1,img2,img3,matches,draw_params)

def orb_flan(im1,im2,lowe_ratio,min_match,display,ransac_th):
	import numpy as np
	import cv2 as cv
	from matplotlib import pyplot as plt

	#Read images
	img1 = cv.imread(im1,1)
	img2 = cv.imread(im2,1)
	#Making sift object - it takes default but also can take
	#(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma)
	orb = cv.ORB_create()
	#Exporting to grayscale
	gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
	gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	#Detecting keypoints and computing descriptors for every keypoint
	kp1, des1 = orb.detectAndCompute(gray1,None)
	kp2, des2 = orb.detectAndCompute(gray2,None)

	if (kp1 is None or kp2 is None or des1 is None or des2 is None):
		a,b,c,d = type(kp1), type(kp2), type(des1), type(des2)
		return(a,b,c,d,None)

	if len(des1) < 2 or len(des2) < 2:
		return(None, None, None, None, None)

	#Store all the good matches as per Lowe's ratio test.
	FLANN_INDEX_LSH = 6

	index_params= dict(algorithm = FLANN_INDEX_LSH,
					   table_number = 6, # 12
					   key_size = 12,     # 20
					   multi_probe_level = 1) #2

	search_params = dict(checks = 100)
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)


	if matches is None:
		return(None, None, None, None, None)

	if lowe_ratio == None:
		lowe_ratio = 0.75

	good = []
	# for m,n in matches:
	# 	if m.distance < lowe_ratio*n.distance:
	# 		good.append(m)
	for m_n in matches:
		if len(m_n) != 2:
			continue
		(m,n) = m_n
		if m.distance < 0.6*n.distance:
			good.append(m)


	if min_match == None:
		MIN_MATCH_COUNT = 10
	else:
		MIN_MATCH_COUNT = min_match

	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		if ransac_th== None:
			ransac_th = 5.0

		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,ransac_th)
		matchesMask = mask.ravel().tolist()

		h,w,d = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv.perspectiveTransform(pts,M)

		img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)
	img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	if display != None:
			plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB)),plt.axis("off"),plt.show()
	return(img1,img2,img3,good,draw_params)
