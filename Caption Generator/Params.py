import torch
from torch.distributions import Categorical

class Params:

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Architecture
    emb_dim = 24  # same as Google paper (512)
    batch_size = 100
    max_pred_sen = 20
    hidden_size = 12
    n_layers = 1

    # Training
    n_epochs = 20
    lr = 3e-4
    dropout_rate = 0.4

    bleu_ngram = 2

    def sampling_strat(self, input, top_k_val: int = 3):
        """ Categorical sampling from the input vector
            Input:
              input: (1,vocab_size)
            Output: index of sampled token
        """
        top_k = torch.topk(input, top_k_val, dim= 1)
        local_idx = Categorical(logits= top_k.values).sample()
        return top_k.indices[0,local_idx.item()]
         
