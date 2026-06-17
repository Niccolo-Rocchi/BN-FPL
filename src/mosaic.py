from __future__ import annotations

import numpy as np
import pyagrum as gum

from src.config import safe_assert
from src.utils import (get_bn_counts, get_cpt_index, get_cpt_shape,
                       get_min_max_bns, get_tabular_cpt, learn_bn_params,
                       vac_cn)


# Custom CN class
class CN:
    def __init__(self, bn_min_max: tuple = None, cn: gum.CredalNet = None):

        if bn_min_max:
            bn_min, bn_max = bn_min_max
            cn = gum.CredalNet(bn_min, bn_max)
            cn.intervalToCredal()

        elif cn:
            bn_min, bn_max = get_min_max_bns(cn)

        else:
            raise RuntimeError(
                "Please provide either a pair of BNs or a gum.CredalNet."
            )

        self.cpts = dict()
        for var in bn_min.names():
            shape = get_cpt_shape(bn_min.cpt(var))
            cpt_min_max = (bn_min.cpt(var), bn_max.cpt(var))
            self.cpts[var] = CN_CPT(var, shape, cpt_min_max)

        self.bn_min = bn_min
        self.bn_max = bn_max
        self.cn = cn

    def names(self):

        return self.cn.current_bn().names()

    def get(self):
        return self.bn_min, self.bn_max

    def cpt_min(self, var: str):

        return self.cpts[var].cpt_min

    def cpt_max(self, var: str):

        return self.cpts[var].cpt_max

    def cpt(self, var):

        return get_tabular_cpt(self.cpt_min(var)), get_tabular_cpt(self.cpt_max(var))

    def update_cpt(self, var, cpt_min_max: tuple):

        # Update the CPT
        self.cpts[var].update(cpt_min_max)

        # Update the global CN
        new_cpt_min, new_cpt_max = self.cpts[var].get()

        self.bn_min.cpt(var).fillWith(new_cpt_min.flatten())
        self.bn_max.cpt(var).fillWith(new_cpt_max.flatten())

    def __str__(self):

        str_base = f"# Object: {self.__class__.__name__}\n\n## Variables:"
        for var in self.names():
            str_var = "\n\n### " + str(self.cpts[var])
            str_base += f"{str_var}"

        return str_base


# Custom CN's CPT class
class CN_CPT:

    def __init__(self, var: str, shape: tuple, cpt_min_max: tuple = None):

        if cpt_min_max:
            cpt_min, cpt_max = cpt_min_max
            self.vacuous = False
        else:
            cpt_min, cpt_max = np.zeros(shape), np.ones(shape)
            self.vacuous = True

        self.var = var
        self.shape = shape
        self.cpt_min = get_tabular_cpt(cpt_min)
        self.cpt_max = get_tabular_cpt(cpt_max)

    def update(self, cpt_min_max: tuple):
        cpt_min, cpt_max = cpt_min_max
        self.vacuous = False

        self.cpt_min = get_tabular_cpt(cpt_min)
        self.cpt_max = get_tabular_cpt(cpt_max)

    def get(self):
        return self.cpt_min, self.cpt_max

    def __str__(self):

        name = self.__class__.__name__
        vacuous_str = f"(vacuous: {self.vacuous})" if name == "PriorCPT" else ""

        return f"Object: {name} for variable {self.var} {vacuous_str} \n\n CPT min:\n {str(self.cpt_min)}\n\n CPT max:\n {str(self.cpt_max)}"


# Prior for a CPT
class PriorCPT(CN_CPT):

    def __init__(self, var: str, shape: tuple):

        super().__init__(var, shape)

        self.clients = None
        self.method = None
        self.weighting = None

    def set_clients(self, clients: list):
        self.clients = clients

    def is_vacuous(self) -> bool:

        return self.vacuous

    def compute(self, method: str, weighting: str = None) -> tuple:

        if self.vacuous:

            self.method = method
            self.weighting = weighting

            if len(self.clients) == 1:
                new_cpt_min, new_cpt_max = self.clients[0].get_cset(self.var)

            else:
                cpts = np.zeros((2, *self.shape, len(self.clients)))
                for i in range(len(self.clients)):
                    client = self.clients[i]
                    cpt_min, cpt_max = client.get_cset(self.var)
                    cpts[0, ..., i] = get_tabular_cpt(cpt_min)
                    cpts[1, ..., i] = get_tabular_cpt(cpt_max)

                if self.method == "ConvComb":

                    if self.weighting == "unif":
                        w = 1 / len(self.clients)
                        cpts_sum = np.sum(cpts, axis=-1) * w
                        new_cpt_min, new_cpt_max = cpts_sum[0, ...], cpts_sum[1, ...]

                        safe_assert(np.all(new_cpt_min <= new_cpt_max + 1e-6))
            return new_cpt_min, new_cpt_max

        else:
            return self.get()


# Prior for a CN
class PriorCN(CN):

    def __init__(self, bn: gum.BayesNet):

        super().__init__(cn=vac_cn(bn))

        self.cpts = dict()
        for var in bn.names():
            shape = get_cpt_shape(bn.cpt(var))
            self.cpts[var] = PriorCPT(var, shape)

    def get_mask(self, var: str) -> np.array:
        if self.cpts[var].clients is None:
            raise RuntimeError(f"Clients are not set for variable {var}.")

        clients = self.cpts[var].clients
        mask = get_tabular_cpt(clients[0].mask.cpt(var))

        if len(clients) == 0:
            pass
        else:
            for c in clients[1:]:
                mask *= get_tabular_cpt(c.mask.cpt(var))

        return mask

    def is_vacuous_cpt(self, var: str):
        return self.cpts[var].is_vacuous()

    def is_vacuous_any(self):

        for var in self.names():
            if self.is_vacuous_cpt(var):
                return True
        return False

    def is_vacuous_all(self):

        for var in self.names():
            if not self.is_vacuous_cpt(var):
                return False
        return True

    # Compute a prior CPT
    def compute_cpt(self, var: str, clients: list, method: str, weighting: str = None):

        # Compute the CPT
        self.cpts[var].set_clients(clients)
        result = self.cpts[var].compute(method, weighting)

        # Update the CPT
        self.update_cpt(var, result)

    # Compute all prior CPTs
    def compute(self, clients: list, method: str, weighting: str = None):

        for var in self.names():
            self.compute_cpt(var, clients, method, weighting)


# Class for a client
class Client:
    n_clients = 0

    def __init__(self, gt: gum.BayesNet, mask: gum.BayesNet):
        self.label = Client.n_clients
        Client.n_clients += 1

        # Set the client's ground-truth BN and mask w.r.t. baseline BN
        self.gt = gt
        self.mask = mask

        # Set the prior CN
        self.prior_cn = PriorCN(self.gt)

        # Set the mosaic CN to vacuous (default)
        self.cn_mosaic = CN(cn=vac_cn(self.gt))

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
        """
        Generate client's data based on self.gt.
        """

        gen = gum.BNDatabaseGenerator(self.gt)
        gen.drawSamples(size)
        self.data = gen.to_pandas()

    def learn_bn(self) -> None:
        """
        Learn the BN parameters by MLE, provided the DAG structure.
        """
        self.check(["data"])
        self.bn = learn_bn_params(self.gt, self.data)

    def learn_cn(self, ess: int) -> None:
        """
        Learn the CN by local IDM, provided the DAG structure.
        """
        self.check(["data", "bn"])
        self.bn_counts = get_bn_counts(self.bn, self.data)
        cn = gum.CredalNet(self.bn_counts)
        cn.idmLearning(ess)

        self.ess = ess
        self.cn = CN(cn=cn)

    def get_cset(self, var: str, parents: dict = None) -> tuple:
        """
        Get the credal set `var`|`parents`, .
        If `parents` is None, return all csets.
        """

        self.check(["cn"])

        bn_min, bn_max = self.cn.bn_min, self.cn.bn_max
        cpt_min, cpt_max = bn_min.cpt(var)[:], bn_max.cpt(var)[:]
        cpt_min, cpt_max = get_tabular_cpt(cpt_min), get_tabular_cpt(cpt_max)

        if parents is None and len(self.gt.parents(var)) != 0:
            return cpt_min, cpt_max

        parents_idx = get_cpt_index(self.gt, var, parents)
        row_min = cpt_min[parents_idx, :]
        row_max = cpt_max[parents_idx, :]

        return row_min, row_max

    def mosaic_cn_cpt(self, var: str):
        """
        Update the `var`'s CPT by MOSAIC.
        """

        self.check(["cn", "bn", "bn_counts"])

        prior_cpt_min, prior_cpt_max = self.prior_cn.cpt(var)

        cpt_mle = get_tabular_cpt(self.bn.cpt(var))
        cpt_counts = get_tabular_cpt(self.bn_counts.cpt(var))
        ess = self.ess

        new_cpts = np.zeros((2, *cpt_mle.shape))

        index = self.bn.cpt(var).topandas().index
        index_df = index.to_frame(index=False)

        # For every parent configuration ...
        parents_conf = (
            [dict(index_df.iloc[i]) for i in range(len(index_df))]
            if len(self.bn.parents(var)) != 0
            else [None]
        )
        for parents in parents_conf:

            parents_idx = get_cpt_index(self.bn, var, parents)

            # ... get the prior, ...
            prior_min, prior_max = (
                prior_cpt_min[parents_idx, :],
                prior_cpt_max[parents_idx, :],
            )

            # ... get the MLE and counts, ...
            row_mle = cpt_mle[parents_idx, :]
            row_counts = cpt_counts[parents_idx, :]

            # ... and update by MOSAIC
            n_pi = np.sum(row_counts)
            new_row_min = n_pi / (n_pi + ess) * row_mle + ess / (n_pi + ess) * prior_min
            new_row_max = n_pi / (n_pi + ess) * row_mle + ess / (n_pi + ess) * prior_max

            new_cpts[0, parents_idx, :] = new_row_min
            new_cpts[1, parents_idx, :] = new_row_max

        new_cpt_min = np.squeeze(new_cpts[0, ...])
        new_cpt_max = np.squeeze(new_cpts[1, ...])

        # Debug
        cpt_min, cpt_max = self.get_cset(var)
        assert np.all(new_cpt_min >= cpt_min - 1e-6)
        assert np.all(new_cpt_max <= cpt_max + 1e-6)
        assert np.all(prior_cpt_min <= prior_cpt_max + 1e-6)
        assert prior_cpt_min.shape == prior_cpt_max.shape

        self.cn_mosaic.update_cpt(var, (new_cpt_min, new_cpt_max))

    def mosaic_cn(self):
        for var in self.bn.names():
            self.mosaic_cn_cpt(var)
