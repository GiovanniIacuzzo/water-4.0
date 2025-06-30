import torch
import torch.nn as nn

# === UTILI ===
def get_embedding_sizes(num_categories, dim_per_category):
    return [(n, d) for n, d in zip(num_categories, dim_per_category)]

class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dims, hidden_dim=128):
        super(Generator, self).__init__()
        input_dim = noise_dim + sum(cond_dims)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Output: [duration, severity]
        )

    def forward(self, z, cond_vec):
        x = torch.cat([z, cond_vec], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, cond_dims, hidden_dim=128):
        super(Discriminator, self).__init__()
        input_dim = 2 + sum(cond_dims)  # 2 = duration + severity

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond_vec):
        x = torch.cat([x, cond_vec], dim=1)
        return self.net(x)


class CondEmbedding(nn.Module):
    """
    Embedding multiplo per feature categoriali condizionali:
    weekday (7), month (12), leak_type (3), start_time (24)
    """
    def __init__(self):
        super(CondEmbedding, self).__init__()
        
        self.emb_weekday = nn.Embedding(7, 3)
        self.emb_month = nn.Embedding(12, 3)
        self.emb_leak_type = nn.Embedding(3, 2)
        self.emb_start_time = nn.Embedding(24, 4)

    def forward(self, weekday, month, leak_type, start_time):
        emb1 = self.emb_weekday(weekday)
        emb2 = self.emb_month(month)
        emb3 = self.emb_leak_type(leak_type)
        emb4 = self.emb_start_time(start_time)
        return torch.cat([emb1, emb2, emb3, emb4], dim=1)

    def output_dim(self):
        return 3 + 3 + 2 + 4  # = 12


def build_models(noise_dim=16):
    cond_embedder = CondEmbedding()
    cond_dim = cond_embedder.output_dim()

    G = Generator(noise_dim, [cond_dim])
    D = Discriminator([cond_dim])

    return G, D, cond_embedder
