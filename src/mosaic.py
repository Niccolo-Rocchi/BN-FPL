from __future__ import annotations

import copy
import sys
import warnings

import numpy as np
import pyagrum as gum
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config import safe_assert
from src.utils import (
    check_intersection,
    get_bn_counts,
    get_cpt_index,
    get_cpt_shape,
    get_min_max_bns,
    get_parent_confs,
    get_tabular_cpt,
    jsd,
    learn_bn_params,
    mle_bn_from_counts,
    vac_cn,
    vertices_cset,
)


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

        self.bn_min.setProperty("name", "bn_min")
        self.bn_max.setProperty("name", "bn_max")

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ["bn_min", "bn_max"]:
                if getattr(self, k) is None:
                    setattr(new, k, None)
                else:
                    v_new = gum.BayesNet(getattr(self, k))
                    setattr(new, k, v_new)
            elif k in ["cn"]:
                v_new = gum.CredalNet(self.bn_min, self.bn_max)
                setattr(new, k, v_new)
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new

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

        self.compute_vertices()

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new

    def compute_vertices(self):
        v_list = []
        for idx in range(len(self.cpt_min)):
            cset_min = self.cpt_min[idx, :]
            cset_max = self.cpt_max[idx, :]

            vertices = vertices_cset(cset_min, cset_max)
            v_list.append(vertices)

        self.vertices = v_list

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
        self.weighting = None
        self.intersection_matrix = None
        self.intersection_frac = None

    def set_clients(self, clients_list: list):
        """
        For convenience: clients_list[0] is the client which `self` belongs;
        clients_list[1:] are the clients to use as prior.
        """
        self.clients = [copy.deepcopy(c) for c in clients_list]

    def is_vacuous(self) -> bool:

        return self.vacuous

    def compute(self, weighting: int) -> tuple:
        """
        Compute a convex combination of prior clients' CPTs, using weighting
        scheme `weighting` (see Table `weight_schemas` in cap6_extract.tex):

          1 = equal contributions, no filtering (#1: w_i propto 1)
          2 = equal contributions, restricted to candidates whose credal set
              intersects the target's -- a hard, boolean cutoff (#4:
              w_i propto I(K^e cap K^i))
          3 = equal contributions, weighted by proximity to the target's own
              MLE theta_hat^e via JSD -- a soft cutoff that never assigns
              zero weight (replaces #5, w_i propto mu(K^e cap K^i), which
              collapses to 0 on empty intersection and, more importantly,
              rewards wide/imprecise candidates simply because they overlap
              more in absolute terms). For each candidate i:
                  d_i = max_{v in vertices(K^i)} JSD(theta_hat^e, v)
              attained at a vertex of K^i since JSD(theta_hat^e, .) is convex
              (Lin 1991). Since JSD is bounded by ln(2), w_i propto ln(2)-d_i
              is always strictly positive in practice, and correctly favors
              candidates that are both close to theta_hat^e AND precise
              (narrow K^i), instead of conflating overlap with imprecision.

        Rows (parent configurations) for which no candidate client contributes
        to the prior -- either because there are no candidate clients at all,
        or because weighting=2 and none of them intersects `self` -- are left
        vacuous, i.e. the prior is [0, 1] there (Eq. `s_oper`, S(\\emptyset) =
        \\Delta_s), meaning no update takes place for that row.
        """

        self.weighting = weighting
        n_candidates = len(self.clients) - 1

        # Extract the prior clients' CPTs related to self.var
        cpts = np.zeros((2, *self.shape, n_candidates))
        for i in range(n_candidates):
            c = self.clients[i + 1]
            cpt_min, cpt_max = c.get_cset(self.var)
            cpts[0, ..., i] = get_tabular_cpt(cpt_min)
            cpts[1, ..., i] = get_tabular_cpt(cpt_max)

        if weighting == 3:
            target_mle = get_tabular_cpt(self.clients[0].bn_mle.cpt(self.var))

        # Weighted average of prior clients' CPTs
        I = np.ones((self.shape[0], n_candidates))  # weighting factor (scheme-dependent)
        I_bin = np.ones((self.shape[0], n_candidates))  # 1 = intersects (diagnostic, always computed)
        for row in range(I.shape[0]):
            self_vertices = self.clients[0].cn.cpts[self.var].vertices[row]
            for c in range(I.shape[1]):
                c_vertices = self.clients[c + 1].cn.cpts[self.var].vertices[row]
                does_intersect = check_intersection(self_vertices, c_vertices)
                I_bin[row, c] = float(does_intersect)

                if weighting == 2 and not does_intersect:
                    I[row, c] = 0
                elif weighting == 3:
                    d_i = max(jsd(target_mle[row], v) for v in c_vertices)
                    I[row, c] = np.log(2) - d_i

        # Diagnostic: fraction of candidate clients intersecting `self`, per row
        self.intersection_frac = (
            I_bin.mean(axis=-1) if n_candidates > 0 else np.full(self.shape[0], np.nan)
        )

        I_sum = np.sum(I, axis=-1, keepdims=True)
        vacuous_rows = I_sum[:, 0] == 0
        if np.any(vacuous_rows):
            # warnings.warn(
            #     f"No client contributes to the prior for variable {self.var} "
            #     f"in {int(np.sum(vacuous_rows))} parent configuration(s); "
            #     "falling back to a vacuous prior there."
            # )
            pass

        W = np.divide(I, I_sum, out=np.zeros_like(I, dtype=float), where=I_sum != 0)
        cpts_weighted = cpts * W[None, :, None, :]
        cpts_sum = np.sum(cpts_weighted, axis=-1)

        new_cpt_min, new_cpt_max = cpts_sum[0, ...], cpts_sum[1, ...]
        new_cpt_min[vacuous_rows, :] = 0.0
        new_cpt_max[vacuous_rows, :] = 1.0

        # Debug
        non_vacuous = ~vacuous_rows
        safe_assert(
            not np.any(non_vacuous)
            or np.all(np.sum(W[non_vacuous], axis=-1) < 1 + 1e-6)
        )
        safe_assert(np.all(new_cpt_min < new_cpt_max + 1e-6))

        self.intersection_matrix = I
        return new_cpt_min, new_cpt_max


# Prior for a CN
class PriorCN(CN):

    def __init__(self, bn: gum.BayesNet):

        super().__init__(cn=vac_cn(bn))

        self.cpts = dict()
        for var in bn.names():
            shape = get_cpt_shape(bn.cpt(var))
            self.cpts[var] = PriorCPT(var, shape)

    def get_mask(self, var: str) -> np.array:

        clients_list = self.cpts[var].clients

        if clients_list is None:
            raise RuntimeError(f"Clients are not set for variable {var}.")

        mask = get_tabular_cpt(clients_list[1].mask.cpt(var))

        if len(clients_list) == 2:
            pass
        else:
            for c in clients_list[2:]:
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
    def compute_cpt(self, var: str, clients_list: list, weighting: int) -> np.array:

        # Compute the CPT
        self.cpts[var].set_clients(clients_list)
        result = self.cpts[var].compute(weighting)

        # Update the CPT
        self.update_cpt(var, result)

        return self.cpts[var].intersection_frac

    # Compute all prior CPTs. Returns the median (over all variables and parent
    # configurations in the network) fraction of candidate clients whose credal
    # set intersects the target client's one -- a diagnostic of how much genuine
    # overlap is available across the whole network, regardless of `weighting`.
    def compute(self, clients_list: list, weighting: int) -> float:

        fracs = []
        for var in self.names():
            fracs.append(self.compute_cpt(var, clients_list, weighting))
        fracs = np.concatenate(fracs)
        fracs = fracs[~np.isnan(fracs)]

        return float(np.median(fracs)) if len(fracs) > 0 else float("nan")


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
        self.bn_mle = None
        self.bn_counts = None
        self.cn = None
        self.ess = None

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ["gt", "mask", "bn", "bn_mle", "bn_counts"]:
                if getattr(self, k) is None:
                    setattr(new, k, None)
                else:
                    v_new = gum.BayesNet(getattr(self, k))
                    setattr(new, k, v_new)
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new

    @retry(stop=stop_after_attempt(100), wait=wait_fixed(0.05))
    def generate_base_info(self, n, ess):

        # Generate data
        self.generate_data(size=n)

        # Learn the BN
        self.learn_bn()

        # Learn the CN
        self.learn_cn(ess=ess)

    def check(self, attributes: list):
        for a in attributes:
            if getattr(self, a) is None:
                raise RuntimeError(f"Attribute '{a}' is missing.")

    def reset_prior(self):
        self.prior_cn = PriorCN(self.gt)

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
        for name in self.bn.names():
            assert self.bn.variable(name).labels() == self.gt.variable(name).labels()
        self.bn.setProperty("name", "bn")

    def learn_cn(self, ess: int) -> None:
        """
        Learn the CN by local IDM, provided the DAG structure.
        """
        self.check(["data", "bn"])
        self.bn_counts = get_bn_counts(self.bn, self.data)
        # The exact (unsmoothed) MLE, kept separate from `self.bn` -- see
        # `mle_bn_from_counts`. Use this, not `self.bn`, whenever the true
        # theta_hat is needed (evaluation/reference curves, the MOSAIC
        # update rule).
        self.bn_mle = mle_bn_from_counts(self.bn_counts)
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

        self.check(["cn", "bn", "bn_mle", "bn_counts"])

        prior_cpt_min, prior_cpt_max = self.prior_cn.cpt(var)

        # The exact (unsmoothed) empirical MLE (`self.bn_mle`, see
        # `mle_bn_from_counts`), NOT `self.bn` (learned with a small smoothing
        # prior, see `learn_bn_params`). This must match, term by term, the
        # N[x|pi_X]/N[pi_X] used by `idmLearning` for the update rule to
        # satisfy K^{e+} = K^e under a vacuous prior (see the vacuous-prior
        # remark in cap6_extract.tex, Sec. "Local Learning & Update") --
        # `self.bn`'s smoothing would introduce a small but non-zero mismatch.
        cpt_mle = get_tabular_cpt(self.bn_mle.cpt(var))
        cpt_counts = get_tabular_cpt(self.bn_counts.cpt(var))
        ess = self.ess

        new_cpts = np.zeros((2, *cpt_mle.shape))

        # For every parent configuration ...
        for parents in get_parent_confs(self.bn, var):

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
        try:
            safe_assert(np.all(new_cpt_min >= cpt_min - 1e-4))
            safe_assert(np.all(new_cpt_max <= cpt_max + 1e-4))
            safe_assert(np.all(prior_cpt_min <= prior_cpt_max + 1e-4))
            safe_assert(prior_cpt_min.shape == prior_cpt_max.shape)
        except:
            print("CPT min:\n\n", cpt_min)
            print("\nNew CPT min:\n\n", new_cpt_min)
            raise RuntimeError("Assert tests not passed")

        self.cn_mosaic.update_cpt(var, (new_cpt_min, new_cpt_max))

    def mosaic_cn(self):
        for var in self.bn.names():
            self.mosaic_cn_cpt(var)
