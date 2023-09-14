import os
import numpy as np
from torch.utils.data import Dataset
from utils.utils import get_folder_names
from scipy.io import loadmat
from matplotlib.pyplot import imread
from torch.nn.functional import normalize
from data.dimensionality_reduction import DimensionReducer
from data.scaling import DataScaler

def stack_with_pad(data):
    """
    Add padding both sides
    """
    # add padding to the smaller cubes
    max_shape = np.array([cube.shape for cube in data]).max(axis=0)

    padded_data = []
    for i, cube in enumerate(data):
        shape = np.array(cube.shape)
        diff = max_shape - shape
        pad = np.vstack([np.ceil(diff/2), np.floor(diff/2)]).T.astype(int)
        padded_data.append(np.pad(cube, pad, 'constant', constant_values=(0,)))

    return np.stack(padded_data)

def apply_2d_mask_to_3d_array(mask, arr):
	"""_summary_

	Args:
		mask (np.array(bool)): input mask size [n x m]
		arr (np.array): input array size [n x m x k]

	Returns:
		np.array: Masked array size [n x m x k]
	"""

	assert np.array_equal(mask.shape, arr.shape[0:2]), "Mask and input array first two dimensions should be the same"

	arr_masked = arr

	# input array is 2D
	if arr.ndim < 3:
		arr_masked[~mask] = 0
	else:	
		for i in range(arr.shape[2]):
			arr_masked[~mask, i] = 0

	return arr_masked


def load_fat_depth_regression(data_path, reflectance_type="estimated_reflectance_mehami", use_overall_mask=False):
	sample_names = get_folder_names(data_path)
	
	reflectance_cube = []
	fat_depth_map = []
	masks = []
	rgb_projected = []

	# read in all reflectance_cube from samples in directory
	for sample_name in sample_names:
		# get appropriate mask 
		if use_overall_mask:
			mask = imread(os.path.join(data_path, sample_name, 'pixel_masks', f'overall.png'))
			mask = mask.astype(bool)
			mask.append(mask)
		else:
			mask = imread(os.path.join(data_path, sample_name, 'pixel_masks', f'{reflectance_type}.png'))
			mask = mask.astype(bool)
			masks.append(mask)

		# load hypercube from MAT file
		hypercube_sample = loadmat(os.path.join(data_path, sample_name, 'hypercube-wise', f'{reflectance_type}.mat'))

		# load and mask reflectance hypercube
		reflectance_cube.append(apply_2d_mask_to_3d_array(mask, hypercube_sample['reflectanceCubeX']))

		# load and mask fat depth map
		fat_depth_map.append(apply_2d_mask_to_3d_array(mask, hypercube_sample['fatDepthCubeY']))
		
		# load and mask rgb image
		rgb_projected.append(apply_2d_mask_to_3d_array(mask, imread(os.path.join(data_path, sample_name, 'frame_colour_projected_hs.png'))))

	# pad arrays with zeros to be all equal size
	reflectance_cube = stack_with_pad(reflectance_cube)
	fat_depth_map = stack_with_pad(fat_depth_map)
	masks = stack_with_pad(masks)
	rgb_projected = stack_with_pad(rgb_projected)

	fat_depth_data = {
		'reflectance_cube': reflectance_cube,
		'fat_depth_map': fat_depth_map,
		'masks': masks,
		'rgb_projected': rgb_projected,
		'sample_names': sample_names,
		'number_samples': len(sample_names)
	}

	return fat_depth_data


class FatDepthDataset(Dataset):
	def __init__(self, fat_depth_data, scaler, reducer, load_as_image=False):
		self.reflectance_cube = fat_depth_data['reflectance_cube']
		self.fat_depth_map = fat_depth_data['fat_depth_map']
		self.sample_names = fat_depth_data['sample_names']
		self.number_samples = fat_depth_data['number_samples']
		self.masks = fat_depth_data['masks']
		self.load_as_image = load_as_image
		
		# if not training model that requires images (CNN), convert to list of pixel reflectance measurements
		if load_as_image:
			self.data = self.reflectance_cube
			self.labels = self.fat_depth_map
		else:
			curr_mask = self.masks[0,:,:]

			curr_reflectance_image = self.reflectance_cube[0,:,:]
			curr_reflectance_pixels = np.array(curr_reflectance_image[curr_mask].tolist())
			self.data = curr_reflectance_pixels

			curr_fatdepth_image = self.fat_depth_map[0,:,:]
			curr_fatdepth_pixels = np.squeeze(np.array(curr_fatdepth_image[curr_mask].tolist()))
			self.labels = curr_fatdepth_pixels

			for i in range(1,self.number_samples):
				curr_mask = self.masks[i,:,:]

				curr_reflectance_image = self.reflectance_cube[i,:,:]
				curr_reflectance_pixels = np.array(curr_reflectance_image[curr_mask].tolist())
				self.data = np.vstack((self.data, curr_reflectance_pixels))

				curr_fatdepth_image = self.fat_depth_map[i,:,:]
				curr_fatdepth_pixels = np.squeeze(np.array(curr_fatdepth_image[curr_mask].tolist()))
				self.labels = np.hstack((self.labels, curr_fatdepth_pixels))

			# each pixel is a training sample
			self.number_samples = self.labels.shape[0]

		print("Scaling data")
		scaler.fit(self.data)
		self.data = scaler.transform(self.data)

		print("Dimensionality reduction")
		reducer.fit(self.data)
		self.data = reducer.transform(self.data)
		# reducer = DimensionReducer(n_components=20, reduction_method='PCA')
		# reducer.fit(self.data)
		# h = 1
	

	def __getitem__(self, index):
		if self.load_as_image:
			return self.data[index, :, :, :], self.labels[index, :, :]
		
		return self.data[index,:], self.labels[index]
	
	def __len__(self):
		return self.number_samples

			
			

			

			
			

	# 	csv_file = os.path.join(os.curdir, "data", "real_estate_valuation_data_set.csv")
	# 	dataset = pd.read_csv(csv_file, delimiter=",")

	# 	dataset.hist()
	# 	plt.suptitle("Raw Data")
	# 	plt.show(block=False)

	# 	data_transform = StandardizeTransform(dim=0)

	# 	dataset_tensor = torch.from_numpy(dataset.to_numpy(dtype=np.float32)).to(device)
	# 	X = dataset_tensor[:, 0:6]
	# 	self.y = dataset_tensor[:, -1]
	# 	self.X = data_transform(X)
	# 	# self.X = functional.normalize(X, dim=0)
	# 	# self.X = functional.std (X, dim=0)
		

	# 	dataset_normalized = pd.DataFrame(torch.cat((self.X, self.y.reshape(-1,1)), 1).cpu().numpy(), columns=dataset.columns)

	# 	dataset_normalized.hist()
	# 	plt.suptitle("Normalized Data")
	# 	plt.show(block=False)

	# 	self.num_samples = dataset.shape[0]



