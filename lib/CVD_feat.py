import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from multiprocessing import Pool
from matplotlib import pyplot as plt
import time
import os
import cPickle as pickle
import glob
from PIL import Image

TRAINING_FOLDER = '/home/max/CVD/data_train/'
   
def load_cluster_centers(fn):	
    cluster_centers = pickle.load(open(fn, 'rb'))
    return cluster_centers
    
def get_imp(fn):
	return fn.split('/')[-1]
	
def get_ufilenames(path):
    dog_paths = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))
    return dog_paths

def get_filenames(path):
    cat_paths = glob.glob(os.path.join(path, 'c/*.[jJ][pP][gG]'))
    dog_paths = glob.glob(os.path.join(path, 'd/*.[jJ][pP][gG]'))
    return cat_paths, dog_paths

def create_dic(paths):
	d = {}
	for path in paths:
		imp = get_imp(path)
		d[imp] = path
	return d

def extract_desc_pts(fn):
	try:
		img = Image.open(fn).convert('RGB')
	except IOError:
		print fn
		return None
	cv_img = np.array(img)
	cv_img = cv_img[:, :, ::-1].copy() 
	cv_img = cv2.resize(cv_img, (100,100))
	gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
	orb = cv2.ORB_create(nfeatures=200)
	key_pts, desc_pts = orb.detectAndCompute(gray, None)
	return desc_pts
	
	
def nearest(c_c, d_c, test_img):
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	c_matches = bf.knnMatch(np.asarray(test_img, np.float32), np.asarray(c_c,np.float32))
	c_matches = sorted(c_matches, key = lambda x:x.distance)
	d_sum, c_sum = (0, 0)
	for match in c_matches[:20]:
		c_sum += match
	d_matches = bf.match(np.asarray(test_img, np.float32), np.asarray(d_c,np.float32))
	d_matches = sorted(d_matches, key = lambda x:x.distance)
	for match in d_matches[:20]:
		d_sum += match.distance
	if c_sum < d_sum:
		return 'cat'
	else:
		return 'dog'


def load_ims():
	d_c = load_cluster_centers('/home/max/CVD/c_o200c200s400.pickle')
	c_c = load_cluster_centers('/home/max/CVD/d_o200c200s400.pickle')
	test_fns = get_ufilenames('/home/max/CVD/data_test')
	test_data = []
	for i in range(0,40):
		test_desc = extract_desc_pts(test_fns[i])
		test_data.append(test_desc)
		print nearest(c_c, d_c, test_desc), test_fns[i]

def read_and_compute_SIFT(fn):
	try:
		img = Image.open(fn).convert('RGB')
	except IOError:
		print fn
		return None
	cv_img = np.array(img)
	cv_img = cv_img[:, :, ::-1].copy() 
	cv_img = cv2.resize(cv_img, (100,100))
	gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=100, sigma = 1.7)
	(key_pts, desc_pts) = sift.detectAndCompute(gray, None)
	return desc_pts

def import_images():
	#IMPLEMENT TIMER CUTOFF FR+OR IF FEAT EXT TAKES TOO LONG
	d_feats = {'orb': []}
	c_feats = {'orb': []}
	(cat_paths, dog_paths) = get_filenames(TRAINING_FOLDER)
	cat_train_pts = []
	dog_train_pts = []
	for image_fn in shuffle(dog_paths, n_samples = 400, random_state=0):
		odesc_pts = extract_desc_pts(image_fn)
		try:
			for pt in odesc_pts:
				d_feats['orb'].append(pt)
		except TypeError:
			print image_fn
			continue
	for image_fn in shuffle(cat_paths, n_samples = 400, random_state=0):
		odesc_pts = extract_desc_pts(image_fn)
		try:
			for pt in odesc_pts:
				c_feats['orb'].append(pt)
		except TypeError:
			print image_fn
			continue
	cat_k_means = KMeans(n_jobs=-1, n_clusters=200)
	cat_k_means.fit(c_feats['orb'])
	print 'dog calc'
	dog_k_means = KMeans(n_jobs=-1, n_clusters=200)
	dog_k_means.fit(d_feats['orb'])
	print 'saving....'
	with open('/home/max/CVD/d_o200c200s400.pickle', 'wb') as handle:
		pickle.dump(dog_k_means.cluster_centers_, handle)
	with open('/home/max/CVD/c_o200c200s400.pickle', 'wb') as handle:
		pickle.dump(cat_k_means.cluster_centers_, handle)
	return '\n\n\n DONE   '	

def mod():
	cat, dog = get_filenames(TRAINING_FOLDER)
	new = get_ufilenames('home/max/CVD/data_train/')
	c = create_dic(cat)
	d = create_dic(dog)
	n = create_dic(new)
	good_cat = []
	good_dog = []
	for im in c.keys():
		print im
		try:
			good_cat.append(n[im])
		except KeyError:
			continue
	for im in d.keys():
		try:
			good_dog.append(n[im])
		except KeyError:
			continue
	return (good_cat, good_dog)

def read_and_compute_SURF(fn):
    img = cv2.imread(fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    kp, desc = surf.detect(gray, None, useProvidedKeypoints = False)
    return desc

