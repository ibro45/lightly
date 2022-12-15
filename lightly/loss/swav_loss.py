from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules import SwaVPrototypes

@torch.no_grad()
def sinkhorn(
    out: torch.Tensor, 
    iterations: int = 3, 
    epsilon: float = 0.05,
    gather_distributed: bool = False,
) -> torch.Tensor:
    """Distributed sinkhorn algorithm.

    As outlined in [0] and implemented in [1].
    
    [0]: SwaV, 2020, https://arxiv.org/abs/2006.09882
    [1]: https://github.com/facebookresearch/swav/ 

    Args:
        out:
            Similarity of the features and the SwaV prototypes.
        iterations:
            Number of sinkhorn iterations.
        epsilon:
            Temperature parameter.
        gather_distributed:
            If True then features from all gpus are gathered to calculate the
            soft codes Q. 

    Returns:
        Soft codes Q assigning each feature to a prototype.
    
    """
    world_size = 1
    if gather_distributed and dist.is_initialized():
        world_size = dist.get_world_size()

    # get the exponential matrix and make it sum to 1
    Q = torch.exp(out / epsilon).t()
    sum_Q = torch.sum(Q)
    if world_size > 1:
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    B = Q.shape[1] * world_size

    for _ in range(iterations):
        # normalize rows
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        # normalize columns
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()


class SwaVLoss(nn.Module):
    """Implementation of the SwaV loss.

    Attributes:
        temperature:
            Temperature parameter used for cross entropy calculations.
        sinkhorn_iterations:
            Number of iterations of the sinkhorn algorithm.
        sinkhorn_epsilon:
            Temperature parameter used in the sinkhorn algorithm.
        sinkhorn_gather_distributed:
            If True then features from all gpus are gathered to calculate the
            soft codes in the sinkhorn algorithm. 
    
    """

    def __init__(self,
                 input_dim: int = 128, 
                 n_prototypes: Union[List[int], int] = 3000,
                 temperature: float = 0.1,
                 sinkhorn_iterations: int = 3,
                 sinkhorn_epsilon: float = 0.05,
                 sinkhorn_gather_distributed: bool = False):
        super(SwaVLoss, self).__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_gather_distributed = sinkhorn_gather_distributed
        # Queues are initiliazed in the first forward call.
        self.queues = []

        self.prototypes = SwaVPrototypes(input_dim, n_prototypes)


    def subloss(self, z: torch.Tensor, q: torch.Tensor):
        """Calculates the cross entropy for the SwaV prediction problem.

        Args:
            z:
                Similarity of the features and the SwaV prototypes.
            q:
                Codes obtained from Sinkhorn iterations.

        Returns:
            Cross entropy between predictions z and codes q.

        """
        return - torch.mean(
            torch.sum(q * F.log_softmax(z / self.temperature, dim=1), dim=1)
        )


    def forward(self,
                high_resolution_features: List[torch.Tensor],
                low_resolution_features: List[torch.Tensor]):
        """Assigns the prototypes and computes the SwaV loss
        for a set of high and low resolution features.

        Args:
            high_resolution_features:
                List of features for the high resolution crops.
            low_resolution_features:
                List of features for the low resolution crops.

        Returns:
            Swapping assignments between views loss (SwaV) as described in [0].

        [0]: SwaV, 2020, https://arxiv.org/abs/2006.09882

        """
        n_crops = len(high_resolution_features) + len(low_resolution_features)
        device = high_resolution_features[0].device

        # Create one queue for each high resolution view. Done on the first forward call only
        if not self.queues:
            #device = next(self.parameters()).device
            for _ in range(len(high_resolution_features)):
                queue = MemoryBankModule(3840) ######size=self.queue_length)
                self.queues.append(queue.to(device))
        
        # Assign the prototypes to multi-crop features
        self.prototypes.to(high_resolution_features[0].device)
        self.prototypes.normalize()
        high_resolution_prototypes = [self.prototypes(x) for x in high_resolution_features]
        low_resolution_prototypes = [self.prototypes(x) for x in low_resolution_features]

        # Get queue prototype assignments
        queue_prototypes = []
        with torch.no_grad():
            for queue, features in zip(self.queues, high_resolution_features):
                _, queue_features = queue(features, update=True)
                queue_prototypes.append(self.prototypes(queue_features))

        # multi-crop iterations
        loss = 0.
        for i in range(len(high_resolution_prototypes)):

            #  compute codes of i-th high resolution crop
            with torch.no_grad():
                prototypes = high_resolution_prototypes[i].detach()

                # Append queue prototypes
                if queue_prototypes is not None:
                    prototypes = torch.cat((prototypes, queue_prototypes[i].detach()))

                q = sinkhorn(
                    prototypes,
                    iterations=self.sinkhorn_iterations,
                    epsilon=self.sinkhorn_epsilon,
                    gather_distributed=self.sinkhorn_gather_distributed,
                )

                # Drop queue similarities
                if queue_prototypes is not None:
                    q = q[:len(high_resolution_prototypes[i])]

            # compute subloss for each pair of crops
            subloss = 0.
            for v in range(len(high_resolution_prototypes)):
                if v != i:
                    subloss += self.subloss(high_resolution_prototypes[v], q)

            for v in range(len(low_resolution_prototypes)):
                subloss += self.subloss(low_resolution_prototypes[v], q)

            loss += subloss / (n_crops - 1)

        return loss / len(high_resolution_prototypes)
