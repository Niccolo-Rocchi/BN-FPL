from src.config import set_seed
import pyagrum as gum
import numpy as np

from src.mosaic import Client
from src.utils import resample_bn_params


def test_resample_bn_params():
    set_seed()
    bn_base = gum.fastBN(f"X<-Y->W->Z; X->Z")
    for prob in [0.0, 0.5, 1.0]:

        for _ in range(5):
            
            bn_new, bn_mask = resample_bn_params(bn_base, prob=prob)

            for node_id in bn_base.nodes():
                cpt = bn_base.cpt(node_id)[:]
                cpt_new = bn_new.cpt(node_id)[:]
                cpt_and = np.array(cpt == cpt_new, dtype=int)

                assert(np.allclose(cpt_and - bn_mask.cpt(node_id)[:], 0))



def test_update_cn_cpt():

    set_seed()
    bn_base = gum.fastBN(f"X<-Y->W->Z; X->Z")

    # Generate clients
    clients = {}
    E, N = 2, 100
    for e in range(E):

        # Init the client
        p = 0 if e == 0 else .2
        gt, mask = resample_bn_params(bn_base, prob=p)
        client = Client(gt, mask)

        # Generate data
        client.generate_data(size=N)

        # Learn the BN
        client.learn_bn()

        # Learn CNs
        client.learn_cn(ess=1)

        # Collect
        clients[e] = client


    for e in range(len(clients)):
        c = clients[e]
        c.mosaic_cn()  # vacuous prior

        for var in c.bn.names():
            assert(np.allclose([x.flatten() for x in c.get_cset(var)], [x.flatten() for x in c.cn_mosaic.cpt(var)]))    #TODO: check array shapes, must be consistent
