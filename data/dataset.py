import os
import numpy as np
from torch.utils.data import Dataset
from utils.utils import get_folder_names
from scipy.io import loadmat
from matplotlib.pyplot import imread
from torch.nn.functional import normalize

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



class FatDepthDataset(Dataset):
	def __init__(self, data_path, load_as_image=False, reflectance_type="estimated_reflectance_mehami"):
		self.data_path = data_path
		self.sample_names = get_folder_names(data_path)
		self.load_as_image = load_as_image
		
		self.data = []
		self.labels = []
		self.masks = []
		self.masks_overall = []
		self.rgb_projected = []
	
		# read in all data from samples in directory
		for sample_name in self.sample_names:
			# load hypercube from MAT file
			hypercube_sample = loadmat(os.path.join(data_path, sample_name, 'hypercube-wise', f'{reflectance_type}.mat'))
			self.data.append(hypercube_sample['fatDepthCubeY'])
			self.labels.append(hypercube_sample['reflectanceCubeX'])
			
			mask = imread(os.path.join(data_path, sample_name, 'pixel_masks', f'{reflectance_type}.png'))
			mask = mask.astype(bool)
			self.masks.append(mask)
			
			mask_overall = imread(os.path.join(data_path, sample_name, 'pixel_masks', f'overall.png'))
			mask_overall = mask_overall.astype(bool)
			self.masks_overall.append(mask_overall)
			
			self.rgb_projected.append(imread(os.path.join(data_path, sample_name, 'frame_colour_projected_hs.png')))
			
			

			

			
			

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


	# def __getitem__(self, index):
	# 	return self.X[index,:], self.y[index]
	
	# def __len__(self):
	# 	return self.num_samples
