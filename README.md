# SN-model  
A minimal model of supply networks.  

This repository serves as supplementary materials to the paper "Local topological features of robust supply networks".  
All the results given in the paper can be reproduced using this model.

For the model to function properly, it is essential to download the [mfinder software](https://www.weizmann.ac.il/mcb/UriAlon/download/network-motif-software), compile it for your system and put it into the 'tmp-motifs/mfinder' folder.



## Changes in version 12:
- removed motif score optimization goal function
- added a function to generate a network based on a pool of allowed edges (important when there is a restriction on the edge length)
- EdgeFactory now accepts M_min and M_max as parameters
- reworked the mutation procedure to call basic function of adding edges from the allowed pool
- simplified the procedure of computing motif z-scores with mfinder.
- added a separate script to batch-run optimizations using computational clusters


<details><summary>Changelog of undocumented changes since March 2020</summary>

## version 11
    option to chose the robustness type (edges/nodes): 
        - edges
        - nodes
    option to calculate robustness based on all or active edges/nodes.

## version 10
    Not the KM model anymore.


## version 04
    - procedure to calculate the motif signal strength of a single network

## version 03
    - simulated annealing that optimizes for a target robustness
    - added a separate s_t score threshold. This matters only for simane optimizations.
        In it, the score needs to be below s_t and the robustness is calculated against score increase above th.
    - update simulated annealing to store the global best solution. 
        Such solution prioritizes networks with closer robustness rather than lower score.

## version 02.2
    - GA: a proper fix of networks having the same (s, r)
    now they are discarded while assembling the new_population
    - edge/node robustness hidden under a new key 'key_rob_e_n'
    - simane: remove s_t and fix path

## version 02.1
    - modification of the Laplacian solver.
    Now, rows in L are normalized by the number of outgoing links.
    This ensures correct results using the interative approach.
    Results with matrix exponential do not depend on this normalization.

    - Updates to GA.
    New generations are made based on parents + pareto from the previous.
    Pareto -- networks with rank = 0
    Parents -- best [x% of G_size] from the current generation with rank > 0.

## version 02, big update of GA optimization 
    - ranks, pareto, parents are calculated in separate procedures and the values are stored in self.
    - population is filtered to exclude repeating solutions
    - pareto consists of solutions with distinct (s, r) networks
        this was necessary to avoid unlimited growth of parents set
    - networks with same (s, r) do not influence rank of each other 

</details>
