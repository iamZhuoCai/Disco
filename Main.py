import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import logging
import time as Time
from utility import pad_history, calculate_hit, extract_axis_1
from collections import Counter
from Modules_ori import *
import wandb
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.getLogger().setLevel(logging.INFO)



def parse_args():
    parser = argparse.ArgumentParser(description="Run Disco.")

    parser.add_argument('--epoch', type=int, default=500,

                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='MHMisinfo',
                        help='PolitiFact, GossipCop, MHMisinfo')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=2048*8,
                        help='Batch size.')
    parser.add_argument('--layers', type=int, default=1,
                        help='gru_layers')
    parser.add_argument('--hidden_factor', type=int, default=3072,
                        help='Number of hidden factors')
    parser.add_argument('--timesteps', type=int, default=2000,
                        help='timesteps for diffusion')
    parser.add_argument('--ddim_step', type=int, default=100,
                        help='ddim step')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='Learning rate. PolitiFact, MHMisinfo=0.00005, Gossip: 0.0001')
    parser.add_argument('--l2_decay', type=float, default=0.001,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=1,
                        help='cuda device.')
    parser.add_argument('--dropout_rate', type=float, default=0,
                        help='dropout ')
    parser.add_argument('--w', type=float, default=0.0,
                        help='classifier-free guidance weight. Politi: 0.0, Full: 4.0')
    parser.add_argument('--p', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--residual_coef', type=float, default=0.5,
                        help='residual coef.')
    parser.add_argument('--pref_strength', type=float, default=1,
                        help='pref_strength')
    parser.add_argument('--max_rate', type=float, default=0.4,
                        help='max rate.')
    parser.add_argument('--interations_to_max_rate', type=int, default=10000,
                        help='interations to max rate, Full: 10000')
    parser.add_argument('--null_threshold', type=float, default=3,
                        help='null threshold.')
    parser.add_argument('--disentangle_type', type=str, default='pre_disentangle',
                        help='type of diffuser. pre_disentangle, post_disentangle')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='report frequency')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='type of optimizer.')
    parser.add_argument('--beta_sche', nargs='?', default='linear',
                        help='')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    return parser.parse_args()


args = parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


setup_seed(args.random_seed)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, w):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w

        if args.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
                                              beta_end=self.beta_end)
        elif args.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )).float()

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.init_ddim_variables()

    def init_ddim_variables(self):
        indices = list(range(0, self.timesteps + 1, args.ddim_step))
        self.sub_timesteps = len(indices)
        indices_now = [indices[i] - 1 for i in range(len(indices))]
        indices_now[0] = 0
        self.alphas_cumprod_ddim = self.alphas_cumprod[indices_now]
        self.alphas_cumprod_ddim_prev = F.pad(self.alphas_cumprod_ddim[:-1], (1, 0), value=1.0)
        self.sqrt_recipm1_alphas_cumprod_ddim = torch.sqrt(1. / self.alphas_cumprod_ddim - 1)
        self.posterior_ddim_coef1 = torch.sqrt(self.alphas_cumprod_ddim_prev) - torch.sqrt(
            1. - self.alphas_cumprod_ddim_prev) / self.sqrt_recipm1_alphas_cumprod_ddim
        self.posterior_ddim_coef2 = torch.sqrt(1. - self.alphas_cumprod_ddim_prev) / torch.sqrt(
            1. - self.alphas_cumprod_ddim)

    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def i_sample(self, model_forward, model_forward_negative, x, y, t, t_index):
        # x_start = (1 + self.w) * model_forward(x, y, t) - self.w * model_forward_negative(x, t)
        x_start = model_forward(x, y, t)
        x_t = x
        model_mean = (
                self.posterior_ddim_coef1[t_index] * x_start +
                self.posterior_ddim_coef2[t_index] * x_t
        )
        return model_mean

    @torch.no_grad()
    def sample_from_noise(self, model_forward, model_forward_negative, h):
        x = torch.randn_like(h).to(h.device)
        for n in reversed(range(self.sub_timesteps)):
            step = torch.full((h.shape[0],), n * args.ddim_step, device=h.device, dtype=torch.long)
            x = self.i_sample(model_forward, model_forward_negative, x, h, step, n)
        return x
    


    def p_losses(self, denoise_model, x_start, x_start_neg_item, h_non, h_neg, t, noise=None, loss_type="l2"):
        # 
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100

        # 
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        predicted_x_non = denoise_model(x_noisy, h_non, t)
        predicted_x_neg = denoise_model(x_noisy, h_neg, t)

        x_noisy_neg_item = self.q_sample(x_start=x_start_neg_item, t=t, noise=noise)
        predicted_x_neg_item = denoise_model(x_noisy_neg_item, h_non, t)

        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x_non)
        elif loss_type == 'l2':
            x_start_norm = F.normalize(x_start, p=2, dim=1)
            x_start_neg_item_norm = F.normalize(x_start_neg_item, p=2, dim=1)

            predicted_x_non_norm = F.normalize(predicted_x_non, p=2, dim=1)
            predicted_x_neg_norm = F.normalize(predicted_x_neg, p=2, dim=1)

            predicted_x_neg_item_norm = F.normalize(predicted_x_neg_item, p=2, dim=1)

            dot_product_pos = torch.sum(x_start_norm * predicted_x_non_norm, dim=1)

            dot_product_neg = torch.sum(x_start_norm * predicted_x_neg_norm, dim=1)

            dot_product_neg_item = torch.sum(x_start_neg_item_norm * predicted_x_neg_item_norm, dim=1)

            loss_pos = torch.mean((dot_product_pos - 1) ** 2)
            loss_neg = torch.mean((dot_product_neg - 1) ** 2)
            loss_neg_item = torch.mean((dot_product_neg_item - 1) ** 2)

        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x_non)
        else:
            raise NotImplementedError()

        return loss_pos, loss_neg, loss_neg_item

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_negative, x, h, t, t_index):

        # x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_negative(x, neg, t)
        x_t = x
        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model_forward, model_forward_negative, h):

        x = self.sample_from_noise(model_forward, model_forward_negative, h)

        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        self.item_embeddings_id = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings_id.weight, 0, 1)

        self.residual_coef = nn.Parameter(torch.tensor(0.5))


        self.feature_transformation = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Dropout(0.2),
            nn.Linear(4096, hidden_size),
        ).to(device)


        data_directory = './data/' + args.data
        news_embedding_file = os.path.join(data_directory, 'news_embeddings_llama.npy')

        news_embeddings = np.load(news_embedding_file)
        num_embeddings, embedding_dim = news_embeddings.shape


        news_embeddings = torch.from_numpy(news_embeddings).float().to(device)


        news_embeddings = self.feature_transformation(news_embeddings)

        news_embeddings = self.diagonalize_and_scale(news_embeddings)
        news_embeddings = self.diagonalize_and_scale(news_embeddings)


        self.item_embeddings_text = nn.Embedding(num_embeddings + 1, hidden_size)
        self.item_embeddings = nn.Embedding(num_embeddings + 1, hidden_size)
        # self.item_embeddings.weight.data.copy_(torch.cat([torch.zeros(1, hidden_size).to(device), news_embeddings], 0))


        self.item_embeddings_text.weight.data.copy_(torch.cat([torch.zeros(1, hidden_size).to(device), news_embeddings], 0))

        self.item_embeddings.weight.data.copy_(self.item_embeddings_text.weight)



        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)

        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )

        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size * 2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        if self.diffuser_type == 'mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size)
            )
        elif self.diffuser_type == 'mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                nn.LeakyReLU(0.2),
                nn.LayerNorm(self.hidden_size * 2),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size * 2, self.hidden_size)
            )
        

        self.negative_feature_learner = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
        )

        self.non_negative_feature_learner = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
        )


    def diagonalize_and_scale(self, e, epsilon=1e-5, reg_lambda=1e-6):
        var_e = torch.cov(e.T)
        mean_e = torch.mean(e, axis=0)
        eigvals, eigvecs = torch.linalg.eigh(var_e)
        eigvals = eigvals + epsilon
        D = torch.diag(1.0 / torch.sqrt(eigvals))
        O = eigvecs
        transformed_e = (e - mean_e) @ O @ D
    
        return transformed_e
    

    def _initialize_weights(self):
        # 遍历并初始化 nn.Sequential 中的每一层
        for module in self.diffuser:
            if isinstance(module, nn.Linear):  # 针对 nn.Linear 层
                nn.init.normal_(module.weight)  # Xavier 初始化
                # if module.bias is not None:
                #     nn.init.constant_(module.bias, 0.0)  # 偏置初始化为 0
        for module in self.step_mlp:
            if isinstance(module, nn.Linear):  # 针对 nn.Linear 层
                nn.init.normal_(module.weight)  # Xavier 初始化
                # if module.bias is not None:
                #     nn.init.constant_(module.bias, 0.0)  # 偏置初始化为 0
        for module in self.emb_mlp:
            if isinstance(module, nn.Linear):  # 针对 nn.Linear 层
                nn.init.normal_(module.weight)  # Xavier 初始化
                # if module.bias is not None:
                #     nn.init.constant_(module.bias, 0.0)  # 偏置初始化为 0
        for module in self.diff_mlp:
            if isinstance(module, nn.Linear):  # 针对 nn.Linear 层
                nn.init.normal_(module.weight)  # Xavier 初始化
                # if module.bias is not None:
                #     nn.init.constant_(module.bias, 0.0)  # 偏置初始化为 0
        for module in self.feature_transformation:
            if isinstance(module, nn.Linear):  # 针对 nn.Linear 层
                nn.init.normal_(module.weight)  # Xavier 初始化
        for module in self.negative_feature_learner:
            if isinstance(module, nn.Linear):  # 针对 nn.Linear 层
                nn.init.normal_(module.weight)  # Xavier 初始化
        for module in self.non_negative_feature_learner:
            if isinstance(module, nn.Linear):  # 针对 nn.Linear 层
                nn.init.normal_(module.weight)  # Xavier 初始化

    def forward(self, x, h, step):

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        return res

    def forward_negative(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, args.hidden_factor)] * x.shape[0], dim=0)
        # neg = neg.repeat(x.shape[0], 1)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))

        return res

        # return x

    def cacu_x(self, x):
        x = self.item_embeddings(x)

        return x

    def cacu_h(self, states, item_embeddings, len_states, p):
        # hidden
        inputs_emb = item_embeddings[states]
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        # seq = inputs_emb
        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)

        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)

        h = ff_out[:, - 1, :]


        return h
    
    def cacu_h_neg(self, states, item_embeddings, len_states, p):
        # hidden
        inputs_emb = item_embeddings[states]
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)

        seq *= mask
        h = torch.mean(seq, dim=1)


        return h

    def predict(self, states, len_states, diff, target, negative_expanded_embeddings, P_perp):
        # hidden
        neg_embeddings, non_neg_embeddings = self.feature_disentangle(self.item_embeddings.weight)
        inputs_emb = non_neg_embeddings[states]
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)
        
        seq = seq * mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)

        h = ff_out[:, - 1, :]

        x = diff.sample(self.forward, self.forward_negative, h)

        test_item_emb = self.item_embeddings.weight


        scores = torch.matmul(x, test_item_emb.transpose(0, 1))


        return scores

    
    def feature_disentangle(self, emb):
        negative_feature = self.negative_feature_learner(emb)

        non_negative_feature = self.non_negative_feature_learner(emb)


        return negative_feature, non_negative_feature


def evaluate(model, test_data, true_news, diff, negative_expanded_embeddings, device, log_file, P_perp):
    eval_data = pd.read_pickle(os.path.join(data_directory, test_data))

    batch_size = 100
    total_purchase = 0.0
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    mrr_purchase = [0, 0, 0, 0]
    true_rate = [0, 0, 0, 0]

    seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(
        eval_data['next'].values)

    num_total = len(seq)


    for i in range(num_total // batch_size + 1):
        if (i + 1) * batch_size < len(seq):
            seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1) * batch_size], len_seq[i * batch_size: (i + 1) * batch_size], target[i * batch_size: (i + 1) * batch_size]
        else:
            seq_b, len_seq_b, target_b = seq[i * batch_size:], len_seq[i * batch_size:], target[i * batch_size:]

        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(device)

        prediction = model.predict(states, np.array(len_seq_b), diff, target_b, negative_expanded_embeddings, P_perp)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2 = np.flip(topK, axis=1)
        sorted_list2 = sorted_list2
        calculate_hit(sorted_list2, topk, target_b, hit_purchase, ndcg_purchase, mrr_purchase, true_rate, true_news)

        total_purchase += batch_size

    hr_list = []
    ndcg_list = []
    mrr_list = []


    true_rate_list = []

    header_msg = '{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}'.format(
            'HR@' + str(topk[0]), 'HR@' + str(topk[1]), 'HR@' + str(topk[2]), 'HR@' + str(topk[3]),
            'NDCG@' + str(topk[0]), 'NDCG@' + str(topk[1]), 'NDCG@' + str(topk[2]), 'NDCG@' + str(topk[3]),
            'MRR@' + str(topk[0]), 'MRR@' + str(topk[1]), 'MRR@' + str(topk[2]), 'MRR@' + str(topk[3]),
            'TR@' + str(topk[0]), 'TR@' + str(topk[1]), 'TR@' + str(topk[2]), 'TR@' + str(topk[3]))
    print(header_msg)
    if log_file:
        log_file.write(header_msg + '\n')
        log_file.flush()

    for i in range(len(topk)):
        hr_purchase = hit_purchase[i] / num_total
        ng_purchase = ndcg_purchase[i] / num_total
        mr_purchase = mrr_purchase[i] / num_total
        tr_rate = true_rate[i] / num_total

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)

        true_rate_list.append(tr_rate)

        if i == 1:
            hr_20 = hr_purchase


    results_msg = '{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}'.format(hr_list[0], hr_list[1], hr_list[2], hr_list[3], (ndcg_list[0]), (ndcg_list[1]), (ndcg_list[2]), (ndcg_list[3]), (mrr_list[0]), (mrr_list[1]), (mrr_list[2]), (mrr_list[3]), true_rate_list[0],  true_rate_list[1],
                                                                  true_rate_list[2], true_rate_list[3])
    print(results_msg)
    if log_file:
        log_file.write(results_msg + '\n')
        log_file.flush()


    return hr_20

if __name__ == '__main__':
    #

    # args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items
    topk = [1, 5, 10, 20]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timesteps = args.timesteps

    model = Tenc(args.hidden_factor, item_num, seq_size, args.dropout_rate, args.diffuser_type, device)
    diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)


    model.to(device)


    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))

    credible_items_file = os.path.join(data_directory, 'Credible items.npy')
    credible_items_set = np.load(credible_items_file)
    credible_items_set = credible_items_set[credible_items_set !=0]


    all_items = list(range(item_num + 1))
    uncredible_items = [item for item in all_items if item not in credible_items_set]

    # randomly select 20% uncredible items
    num_uncredible_partial = len(uncredible_items) // 5
    random.shuffle(uncredible_items)
    uncredible_items_partial = uncredible_items[:num_uncredible_partial]

    unknown_items_set = [item for item in all_items if item not in uncredible_items_partial]


    unkonwn_items = [item for item in unknown_items_set if item != 0]

    log_path = './Log/ours/{}/{}.txt'.format(args.data, str(Time.time()))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    log_file = open(log_path, 'w')

    log_file.write("Arguments:\n")
    for arg, value in vars(args).items():
        log_file.write(f"{arg}: {value}\n")
    log_file.write("\n")
    log_file.flush()



    total_step = 0
    hr_max = 0
    best_epoch = 0

    num_rows = train_data.shape[0]
    num_batches = int(num_rows / args.batch_size)

    max_rate = args.max_rate
    interations_to_max_rate = args.interations_to_max_rate
    iterations = 0

    for i in range(args.epoch):
        start_time = Time.time()
        for j in range(num_batches):
            # content disentanglement
            all_uncredible_embeddings, all_preference_embeddings = model.feature_disentangle(model.item_embeddings.weight)

            uncredilbe_embeddings = all_uncredible_embeddings[uncredible_items_partial]

            credible_embeddings = model.item_embeddings.weight[unkonwn_items]

            iterations += 1

            # current selection ratio
            current_rate = min(max_rate, (iterations / interations_to_max_rate) * max_rate)


            unkonwn_uncredible_embeddings = all_uncredible_embeddings[unkonwn_items]




            if current_rate > 0:
                # calculating cosine similarity
                negative_norm = F.normalize(uncredilbe_embeddings, dim=1)
                non_negative_norm = F.normalize(unkonwn_uncredible_embeddings, dim=-1)
                sim_matrix = torch.matmul(negative_norm.cuda(), non_negative_norm.cuda().t())

                num_select = max(1, int(current_rate * uncredilbe_embeddings.shape[0]))

                sim_matrix_flat = torch.mean(sim_matrix, 0)

                if num_select >= sim_matrix_flat.shape[0]:
                    topk_indices = torch.arange(sim_matrix_flat.shape[0], device=sim_matrix.device)
                else:
                    topk_values, topk_indices = torch.topk(sim_matrix_flat, num_select)
    

                non_neg_indices = topk_indices % uncredilbe_embeddings.shape[0]
                selected_indices = torch.unique(non_neg_indices)

    
                # select potential uncredible content items
                new_uncredible_embeddings = unkonwn_uncredible_embeddings[selected_indices]
                
                uncredible_expanded_embeddings = torch.cat([uncredilbe_embeddings, new_uncredible_embeddings], dim=0)
            else:
                uncredible_expanded_embeddings = uncredilbe_embeddings

            batch = train_data.sample(n=args.batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target = list(batch['next'].values())


            target_neg = []
            for index in range(args.batch_size):
                neg = np.random.randint(item_num)
                while neg == target[index]:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)


            optimizer.zero_grad()

            seq = torch.LongTensor([[int(x) for x in sub_seq] for sub_seq in seq])
            len_seq = torch.LongTensor([int(x) for x in len_seq])
            target = torch.LongTensor([int(x) for x in target])
            target_neg = torch.LongTensor([int(x) for x in target_neg])

            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            target_neg = target_neg.to(device)

            x_start = model.cacu_x(target)


            x_start_neg_item = model.cacu_x(target_neg)



            if args.disentangle_type == 'pre_disentangle':
                # pre_disentangment
                h_neg = model.cacu_h_neg(seq, all_uncredible_embeddings, len_seq, args.p)
                h_non = model.cacu_h(seq, all_preference_embeddings, len_seq, args.p)
            elif args.disentangle_type == 'post_disentangle':
                # post_disentangment
                h = model.cacu_h(seq, model.item_embeddings.weight, len_seq, args.p)
                h_neg, h_non = model.feature_disentangle(h)



            # SVD
            U, S, V = torch.linalg.svd(uncredible_expanded_embeddings.t(), full_matrices=False, driver="gesvd")
            r = (S > args.null_threshold).sum().item()

            # U2
            credible_basis = U[:, r:]

            P_perp = credible_basis @ credible_basis.t()

            alpha = args.residual_coef

            # Null space projection
            x_start_proj = x_start @ P_perp

            # Residual connection
            x_start_new = alpha * x_start_proj + (1 - alpha) * x_start


            x_start_r = x_start_new


            n = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long()

            loss_term1, loss_term2, loss_term3 = diff.p_losses(model, x_start_r, x_start_neg_item, h_non, h_neg, n, loss_type='l2')

            # L_Disco
            loss = loss_term1 - loss_term2 + args.pref_strength * (loss_term1 - loss_term3)


            loss.backward()

            optimizer.step()



        # scheduler.step()
        if args.report_epoch:
            if i % 1 == 0:
                epoch_msg = "Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(loss) + 'Train cost: {:2f} s'.format(Time.time() - start_time)

                print(epoch_msg)
                log_file.write(epoch_msg + '\n')
                log_file.flush()

            if (i + 1) % 10 == 0:
                eval_start = Time.time()

                test_msg = '-------------------------- TEST PHRASE -------------------------'
                print(test_msg)
                log_file.write(test_msg + '\n')
                log_file.flush()
                _ = evaluate(model, 'test_data.df', credible_items_set, diff, uncredible_expanded_embeddings, device, log_file, P_perp)
                eval_time_msg = 'Test cost: {:2f} s'.format(Time.time() - eval_start)
                print(eval_time_msg)
                log_file.write(eval_time_msg + '\n')
                log_file.flush()
                separator_msg = '----------------------------------------------------------------'
                print(separator_msg)
                log_file.write(separator_msg + '\n')
                log_file.flush()