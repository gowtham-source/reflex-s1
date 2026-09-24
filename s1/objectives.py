"""Open calibration objective, not a reproduction of proprietary Jev RLCD."""
import torch
from torch.nn import functional as F

def proper_loss(logits, targets, ordinal=False):
    p = logits.float().softmax(-1)
    # Soft-target cross entropy and expected Brier: differ from target-vector
    # squared distance by a constant independent of the prediction.
    loss = -(targets * logits.float().log_softmax(-1)).sum(-1)
    loss = loss + .25 * (p - targets).square().sum(-1)
    if ordinal:
        loss = loss + .25 * (p.cumsum(-1) - targets.cumsum(-1)).square().sum(-1)/(p.shape[-1]-1)
    return loss

def training_loss(out, targets, ordinal=False, rl_weight=.1, compute_cost=.02):
    losses = proper_loss(out['all_logits'], targets[:,None,:], ordinal)
    policy = torch.distributions.Categorical(logits=out['policy'])
    sampled = policy.sample()
    costs = torch.arange(1, losses.shape[1]+1, device=losses.device)*compute_cost
    rewards = -(losses.detach() + costs)
    reward = rewards.gather(1, sampled[:,None]).squeeze(1)
    baseline = (policy.probs.detach()*rewards).sum(-1)
    reinforce = -((reward-baseline).detach()*policy.log_prob(sampled)).mean()
    loss = losses.mean() + .01*out['aux'] + rl_weight*(reinforce-.01*policy.entropy().mean())
    return loss, {'proper':losses.mean().item(), 'policy':reinforce.item(), 'reward':reward.mean().item()}
