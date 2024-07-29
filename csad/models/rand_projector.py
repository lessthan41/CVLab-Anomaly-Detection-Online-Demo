import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from einops import rearrange
import tqdm
import numpy as np

def topk(x, k=10, reverse=True):
    # x: NxC ndarray
    # return: Nxk
    if reverse:
        indices = np.argpartition(x, k)[:, :k]
    else:
        indices = np.argpartition(x, -k)[:, -k:]
    result = x[np.arange(x.shape[0])[:, None], indices]
    return result

class RandProjector(nn.Module):
    """
        usage:
        projector = Projector(in_dim=3, projection_dim=2, num_of_projection=5, bins=4)
        for x in data_loader:
            projector.cal_min_max(x)
        for x in data_loader:
            projector.accumulate_global_feature(x)
        projector.estimate_global_density()
        for x in data_loader:
            projector.predict(x)
    """
    def __init__(self,in_dim, projection_dim, num_of_projection=100, bins=20,save_projected=False):
        super(RandProjector,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_dim = in_dim
        self.projection_dim = projection_dim
        self.num_of_projection = num_of_projection
        self.bins = bins
        self.save_projected = save_projected
        self.projected = dict()
        self.projector = nn.Linear(in_dim, projection_dim*num_of_projection,bias=False).to(self.device)
        # print number of projector parameters
        print('number of projector parameters: ', sum(p.numel() for p in self.projector.parameters() if p.requires_grad))
        self.min = torch.ones(self.num_of_projection*self.projection_dim).to(self.device)*999
        self.max = torch.ones(self.num_of_projection*self.projection_dim).to(self.device)*-999
        
        #self.projectors = [nn.Linear(in_dim, projection_dim,bias=False).cuda() for _ in range(num_of_projection)]
        #self.projectors = [nn.Identity() for _ in range(num_of_projection)]
        self.global_vectors = dict() #torch.zeros(num_of_projection, projection_dim, bins).cuda()
        self.init_weights()
        self.projector.requires_grad_(False)

    def init_weights(self):
        #for projector in self.projectors:
        nn.init.normal_(self.projector.weight, mean=0.0, std=1.0)


    def cal_min_max(self, x, sample_idx=0):
        # x: feature_set_length x in_channels
        # min,max: Np x projection_dim
        projected = self.projector(x) # projected: feature_set_length x (projection_dim*num_of_projection)
        if self.save_projected:
            self.projected[sample_idx] = projected
        
        local_min = torch.min(projected, dim=0)[0] # local_min: (projection_dim*num_of_projection)
        local_max = torch.max(projected, dim=0)[0] # local_max: (projection_dim*num_of_projection)
        self.min = torch.min(torch.stack([self.min, local_min], dim=0), dim=0)[0]
        self.max = torch.max(torch.stack([self.max, local_max], dim=0), dim=0)[0]



    def accumulate_global_feature(self,x, sample_idx=0):
        # x: NxC
        self.global_vectors[sample_idx] = self.get_global_feature(x,normalize=False)


    def estimate_global_density(self):
        self.global_vectors = torch.stack(list(self.global_vectors.values()), dim=0) # BxNpxDxB
        self.density_vectors = torch.sum(self.global_vectors, dim=0)
        self.density_vectors = F.normalize(self.density_vectors, p=2, dim=2)
        # for each projection, fit a multi-dimensional KDE
        self.kdes = [KernelDensity(kernel='gaussian', bandwidth=1.0).fit(self.density_vectors[i].cpu().detach().numpy())\
                      for i in range(self.num_of_projection)]
        
    def get_density_vectors(self):
        return self.density_vectors
    
    def get_global_feature(self, x,normalize=True):
        # Complexity: O(Np*D*B)
        # x: feature_set_length x in_channels
        global_vector = torch.zeros(self.num_of_projection, self.projection_dim, self.bins).to(self.device)
        
        projected = self.projector(x)
        # projected: feature_set_length x (projection_dim*num_of_projection)
        # Now reduce the feature_set_length dimension
        projected = rearrange(projected, 'S D -> D S')
        for i in range(self.num_of_projection*self.projection_dim):
            # for each dimension
            # self.min: Number_of_projections x Projected_dim
            dim_min = self.min[i].item()
            dim_max = self.max[i].item()
            
            # Use torch.histogram to replace the innermost loop
            hist = torch.histc(projected[i,:], bins=self.bins, min=dim_min, max=dim_max)
            
            global_vector[i//self.projection_dim, i%self.projection_dim, :] += hist

        if normalize:
            # Normalize each histogram
            global_vector = F.normalize(global_vector, p=2, dim=2)
        return global_vector
    
    def predict_single_kl(self, x):
        # x: feature_set_length x in_channels
        global_vectors = self.get_global_feature(x, normalize=True) # Number_of_projections x Projected_dim x Bins
        global_vectors = global_vectors
        density_vectors = self.get_density_vectors()
        result = list()
        for i in range(self.num_of_projection):
            # sym_kld = 0.5*torch.nn.KLDivLoss(reduction='batchmean')(global_vectors[i,:,:], density_vectors[i,:,:]) + \
            #             0.5*torch.nn.KLDivLoss(reduction='batchmean')(density_vectors[i,:,:], global_vectors[i,:,:])
            sym_kld = torch.nn.KLDivLoss(reduction='batchmean')(global_vectors[i,:,:], density_vectors[i,:,:])
            result.append(sym_kld.item())
        return np.mean(np.array(result))

    def predict_kl(self, x):
        # x: feature_set_length x in_channels
        result = list()
        for i in range(x.shape[0]):
            result.append(self.predict_single_kl(x[i]))
        return np.array(result)


    def predict(self, x):
        # x: BxNxC
        # return: Bx1
        result = list()
        for i in range(x.shape[0]):
            input_global_vectors = self.get_global_feature(x[i])
            # Normalize or not?
            input_global_vectors = F.normalize(input_global_vectors, p=2, dim=2)
            # input_global_vectors: NpxDxB
            anomaly_score = np.zeros(self.num_of_projection)
            for p in tqdm.tqdm(range(self.num_of_projection), desc='predicting each projection...'):
                scores = self.kdes[p].score_samples(input_global_vectors[p].cpu().detach().numpy()) # Dx1
                anomaly_score[p] = np.mean(scores)
            #anomaly_score = np.mean(anomaly_score)
            result.append(anomaly_score)
        result = np.array(result)
        return {'topk_mean':np.mean(topk(result, k=int(self.projection_dim*0.1), reverse=True), axis=1),
                'std':np.std(result, axis=1),
                'mean':np.mean(result, axis=1),
                'min':np.min(result, axis=1)}
    
if __name__ == "__main__":



    projector = RandProjector(in_dim=512, projection_dim=128, 
                          num_of_projection=20, bins=50,
                          save_projected=True)
    for i in range(100):
        x = torch.randn(10, 512).cuda()
        projector.cal_min_max(x, sample_idx=i)
    x = torch.randn(64*100, 512).cuda()
    for i in tqdm.tqdm(range(100)):
        # 100 images , 64 features of 512 dim for each image
        x = x[64*i:64*(i+1)]
        projector.accumulate_global_feature(x, sample_idx=i)
    projector.estimate_global_density()

    a=projector.predict_single_kl(x[0:64])
    b=projector.predict_single_kl(torch.rand(64,512).cuda())
    bb=projector.predict_single_kl(torch.concat([torch.randn(10,512).cuda(),torch.rand(54,512).cuda()],dim=0))
    bbb=projector.predict_single_kl(torch.concat([torch.randn(20,512).cuda(),torch.rand(44,512).cuda()],dim=0))
    bbbb = projector.predict_single_kl(torch.concat([torch.randn(30,512).cuda(),torch.rand(34,512).cuda()],dim=0))
    c=projector.predict_single_kl(torch.randn(64,512).cuda())
    d = (0.5+torch.randn(64,512).cuda()*0.5) + torch.randn(64,512).cuda()
    d=projector.predict_single_kl(d)
    x = torch.stack([torch.randn(64, 512) for i in range(5)], dim=0).cuda()
    x[1,10:50,:] += 0.5
    x[2,20:30,:] += 0.5
    x[3,20:30,:] += 10
    x[4,20:30,:] = x[3,20:30,:]
    out = projector.predict(x)
    global_vectors = [projector.get_global_feature(x[i]).cpu().detach().numpy() for i in range(5)]
    import sklearn.metrics
    # cal all distances
    global_vectors = np.array(global_vectors)
    dist = sklearn.metrics.pairwise_distances(global_vectors[:,0,0,:], metric='euclidean')
    for i in range(1,20):
        dist += sklearn.metrics.pairwise_distances(global_vectors[:,i,0,:], metric='euclidean')
    import matplotlib.pyplot as plt
    # min_max_scale
    dist = (dist - np.min(dist))/(np.max(dist) - np.min(dist))
    np.fill_diagonal(dist, 1)
    plt.imshow(dist)
    plt.show()
    print(out)
    print()