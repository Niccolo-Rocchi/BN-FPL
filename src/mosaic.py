import pyagrum as gum
import numpy as np

from src.utils import get_bn_counts, get_cpt_index, get_cpt_shape, get_min_max_bns, get_tabular_cpt, learn_bn_params


class Client:
    n_clients = 0

    def __init__(self, gt: gum.BayesNet, mask: gum.BayesNet):
        self.label = Client.n_clients
        Client.n_clients += 1     

        # Set the client's ground-truth BN and mask w.r.t. baseline BN
        self.gt= gt
        self.mask = mask

        # Set other args to None
        self.data = None
        self.bn = None
        self.bn_counts = None
        self.cn = None
        self.ess = None


    def check(self, attributes: list):
        for a in attributes:
            if getattr(self, a) is None:
                raise RuntimeError(f"Attribute '{a}' is missing.")
        

    def generate_data(self, size: int) -> None:
        '''
        Generate client's data based on self.gt.
        '''
    
        gen = gum.BNDatabaseGenerator(self.gt)
        gen.drawSamples(size)
        self.data = gen.to_pandas()
    
    def learn_bn(self) -> None:
        '''
        Learn the BN parameters by MLE, provided the DAG structure.
        '''
        self.check(["data"])
        self.bn = learn_bn_params(self.gt, self.data)
        

    def learn_cn(self, ess: int) -> None:
        '''
        Learn the CN by local IDM, provided the DAG structure.
        '''
        self.check(["data", "bn"])
        self.bn_counts = get_bn_counts(self.bn, self.data)
        cn = gum.CredalNet(self.bn_counts)
        cn.idmLearning(ess)

        self.ess = ess
        self.cn = cn
        
    def get_cset(self, var:str, parents:dict = None) -> np.array:
        ''' 
        Get the credal set `var`|`parents`, .
        If `parents` is None, return all csets.
        '''

        self.check(["cn"])
        
        bn_min, bn_max = get_min_max_bns(self.cn)
        cpt_min, cpt_max = bn_min.cpt(var)[:], bn_max.cpt(var)[:]
        cpt_min, cpt_max = get_tabular_cpt(cpt_min), get_tabular_cpt(cpt_max)        

        if parents is None and len(self.gt.parents(var)) != 0:
            return cpt_min, cpt_max
                
        parents_idx = get_cpt_index(self.gt, var, parents)
        row_min = cpt_min[parents_idx,:]
        row_max = cpt_max[parents_idx,:]
        
        return row_min, row_max


    def update_cset(self, var:str, prior: tuple = None, parents:dict = None) -> tuple:

        ''' 
        Update the credal set `var`|`parents`, given a `prior`.
        If `parents` is None, update all csets.
        `prior` is a pair of (prior_min, prior_max).
        '''

        self.check(["cn", "bn", "bn_counts"])

        # No prior means no update
        if prior is None: return self.get_cset(var)
        
        prior_min = prior[0]
        prior_max = prior[1]
        
        cpt_mle = get_tabular_cpt(self.bn.cpt(var))
        cpt_counts = get_tabular_cpt(self.bn_counts.cpt(var))
        ess = self.ess

        if parents and (prior_min.ndim != 1 or prior_max.ndim != 1): 
            raise RuntimeError(f"Please provide a prior of right shape. Actual shape: {prior_min.shape}")
        
        if parents or len(self.bn.parents(var)) == 0: 
            parents_idx = get_cpt_index(self.bn, var, parents)
            row_mle = cpt_mle[parents_idx, :]
            row_counts = cpt_counts[parents_idx, :]

            n_pi = np.sum(row_counts)

            new_row_min = n_pi/(n_pi + ess)*row_mle + ess/(n_pi + ess)*prior_min
            new_row_max = n_pi/(n_pi + ess)*row_mle + ess/(n_pi + ess)*prior_max
        
            # Debug
            row_min, row_max = self.get_cset(var, parents)
            assert(np.all(new_row_min >= row_min - 1e-6))
            assert(np.all(new_row_max <= row_max + 1e-6))
            assert(np.all(prior_min <= prior_max))
            assert(prior_min.shape == prior_max.shape)

            return new_row_min, new_row_max

        else:

            prior_min = get_tabular_cpt(prior[0])
            prior_max = get_tabular_cpt(prior[1])
            
            new_cpt = np.zeros((2,*cpt_mle.shape))

            index = self.bn.cpt(var).topandas().index
            index_df = index.to_frame(index=False)

            for i in range(len(index_df)):

                parents = dict(index_df.iloc[i])
                parents_idx = get_cpt_index(self.bn, var, parents)
                new_row_min, new_row_max = self.update_cset(var, (prior_min[parents_idx,:], prior_max[parents_idx,:]), parents)
                new_cpt[0, i, :] = new_row_min
                new_cpt[1, i, :] = new_row_max

            new_cpt_min = np.squeeze(new_cpt[0, ...])
            new_cpt_max = np.squeeze(new_cpt[1, ...])

            return new_cpt_min, new_cpt_max
        


class CsetFusion():

    def __init__(self, var:str, method: str, weighting: str = None):
        self.var = var
        self.method = method
        self.weighting = weighting

    def __call__(self, clients: list, parents:dict = None):

        if len(clients) == 1: 
            return clients[0].get_cset(self.var)
        
        # Get the CPTs
        shape = get_cpt_shape(clients[0].gt.cpt(self.var))
        cpts = np.zeros((2, *shape, len(clients)))
        for i in range(len(clients)):
            client = clients[i]
            cpt_min, cpt_max = client.get_cset(self.var)
            cpts[0, ..., i] = get_tabular_cpt(cpt_min)
            cpts[1, ..., i] = get_tabular_cpt(cpt_max)

        if self.method == "ConvComb":

            if self.weighting == "unif":
                w = 1/len(clients)
                cpts_sum = np.sum(cpts, axis=-1) * w
                new_cpt_min, new_cpt_max = cpts_sum[0, ...], cpts_sum[1, ...]

                if parents is None:
                    return new_cpt_min, new_cpt_max
                else:
                    parents_idx = get_cpt_index(self.gt, self.var, parents)
                    return new_cpt_min[parents_idx,:], new_cpt_max[parents_idx,:]


            elif self.weighting is None:
                RuntimeError(f"Attribute 'weighting' is missing.")

            else:
                raise NotImplementedError(f"No '{self.weighting}' weighting implemented.")
            
        else:
            raise NotImplementedError(f"No '{self.method}' method implemented.")