import torch
from torch import nn
import torch.nn.functional as F
from models.multihead_attention import MultiHeadAttention
import math
from models.vit import VisionTransformer
from transformers import ViTModel


class Model(nn.Module):
    def __init__(self, args, config_dict):
        super(Model, self).__init__()
        self.args = args
        """
        self.encoder = ViT(
            image_size = 32,
            patch_size = args.vit_patch_size,
            num_classes = num_class,
            dim = 1024,
            depth = args.depth,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        """
        if args.use_inlay==0:
            self.encoder = VisionTransformer(**config_dict)
        else:
            self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        self.head = nn.Linear(768, args.number_id)
        
        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        
        # Why it is the value of k
        self.k = ((32*14 // self.patch_size) - 1)**2

        self.z_size_new = 3 * self.patch_size * self.patch_size

        if self.args.train_value == 1:
            self.z_memory_key_list = nn.Parameter(torch.randn(self.k, self.z_size_new),
                                                  requires_grad=True)
        else:
            self.z_memory_key_list = nn.Parameter(torch.randn(self.k, self.z_size_new) * args.std,
                                                  requires_grad=False)

        # update head num 32 -> 28
        self.multihead_attention = MultiHeadAttention(args=args, in_features=self.z_size_new, head_num=28,
                                                      activation=None, seq_len=self.k)

        self.gamma = nn.Parameter(torch.ones(self.z_size_new))
        self.beta = nn.Parameter(torch.zeros(self.z_size_new))

        self.gamma_key = nn.Parameter(torch.ones(self.z_size_new))
        self.beta_key = nn.Parameter(torch.zeros(self.z_size_new))

    def forward(self, x):
        batch_size = x.shape[0]
        x_org = x

        u = self._divide_patch(x)
        # result: 4 x 3025 x 192
        
        u = self._z_meta_attention(u)
        u = u.transpose(1, 2)
        x = F.fold(u, x.shape[-2:], kernel_size=self.patch_size, stride=self.stride, padding=0)
        
        if self.args.use_inlay==0:
            y_pred_linear = self.head(self.encoder(x))
        else:
            y_pred_linear = self.head(self.encoder(x).last_hidden_state[:, 0, ])
        return y_pred_linear

    def _divide_patch(self, x):
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.stride, padding=0)
        x = x.transpose(1, 2)
        return x

    def _z_meta_attention(self, z):
        """
        :param z: of size (batch, seq, A)
        :return: of size (batch, seq, A)
        """
        batch_size, seq_len, A = z.shape[0], z.shape[1], z.shape[2]

        if self.args.norm_type == 'contextnorm':
            z = self.apply_context_norm(z, gamma=self.gamma, beta=self.beta)

        M_key_batch = torch.stack([self.z_memory_key_list] * batch_size, dim=0)  # (batch, num_memory, A)
        z, _ = self.multihead_attention(query=z, key=z, value=M_key_batch)

        if self.args.norm_type == 'contextnorm':
            z = self.apply_context_norm(z, gamma=self.gamma_key, beta=self.beta_key)
        return z

    def apply_context_norm(self, z_seq, gamma, beta):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * gamma) + beta
        return z_seq
