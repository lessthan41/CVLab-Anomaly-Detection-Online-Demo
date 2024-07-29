import torch
import numpy as np
import torch.nn.functional as F

def kl_divergence(tensor1, tensor2):
	"""
		tensor1: torch.size([1, 1024])
		tensor2: torch.size([8, 1024])
	"""
	kl_divergences = []
	for i in range(tensor2.size(0)):
		p = tensor1.flatten()
		q = tensor2[i].flatten()
		kl_divergences.append(F.kl_div(F.log_softmax(p, dim=0), F.softmax(q, dim=0), reduction='sum').item())
	return kl_divergences

### TODO: Implement Wasserstein distance
def wasserstein_distance(tensor1, tensor2):
	# Assuming tensor1 and tensor2 are torch tensors of shape torch.Size([1764, 1536, 3, 3])
	hw = tensor1.size()[0]
		
	# Reshape tensors to merge the first two dimensions
	merged_tensor1 = tensor1.view(hw, -1)
	merged_tensor2 = tensor2.view(hw, -1)
	# print(merged_tensor1.size())

	# Calculate Wasserstein distance for each feature point in the first channel
	wasserstein_distances = []
	for i in range(merged_tensor1.size(0)):
		p = merged_tensor1[i].flatten()
		q = merged_tensor2[i].flatten()
		sorted_p = torch.sort(p).values
		sorted_q = torch.sort(q).values
		diff = sorted_p - sorted_q
		wasserstein_distance_i = torch.sum(torch.abs(diff))
		wasserstein_distances.append(wasserstein_distance_i.item())

	return wasserstein_distances

# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
