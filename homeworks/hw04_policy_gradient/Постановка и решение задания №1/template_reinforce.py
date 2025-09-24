import numpy as np
import torch
import torch.nn as nn

def to_one_hot(y_tensor, ndims):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def predict_probs(states):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :returns: numpy array of shape [batch, n_actions]
    """
    # convert states, compute logits, use softmax to get probability

    # YOUR CODE GOES HERE
    with torch.no_grad():
        states = torch.FloatTensor(states)
        logits = model(states)
        probs = nn.functional.softmax(logits, dim=-1).numpy()

    return probs

def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    Take a list of immediate rewards r(s,a) for the whole session
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).

    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    A simple way to compute cumulative rewards is to iterate from the last
    to the first timestep and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    # YOUR CODE GOES HERE
    cumulative_rewards = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        cumulative_rewards.append(g)
    cumulative_rewards = cumulative_rewards[::-1]

    return np.array(cumulative_rewards)

def get_loss(states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    """
    Compute the loss for the REINFORCE algorithm.
    """
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int32)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    # predict logits, probas and log-probas using an agent.
    logits = model(states)
    

    probs = torch.softmax(logits, dim=-1)
    

    log_probs = torch.log_softmax(logits, dim=-1)
    

    assert all(isinstance(v, torch.Tensor) for v in [logits, probs, log_probs]), \
        "please use compute using torch tensors and don't use predict_probs function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    log_probs_for_actions = torch.sum(log_probs * to_one_hot(actions, ndims=probs.shape[1]), dim=1) # [batch,]
    
    J_hat = torch.mean(log_probs_for_actions * cumulative_returns)  # a number
    
    
    # Compute loss here. Don't forget entropy regularization with `entropy_coef`
    entropy = - (probs * log_probs).sum(dim=-1).mean()
    
    loss = -J_hat - entropy_coef * entropy
    

    return loss