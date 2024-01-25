from torch.nn import BatchNorm2d
from copy import deepcopy
from fedavg_V1 import FedAvgClient
from collections import OrderedDict
import torch
from src.client.CKA import kernel_CKA


class FedADPlient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.personal_params_name = []
        self.model_client = deepcopy(model.to(self.device))
        self.model_global = deepcopy(model.to(self.device))

    def voting_personal_layer(self,
                              local_parameters: OrderedDict[str, torch.Tensor],
                              global_parameters: OrderedDict[str, torch.Tensor],
                              params_name=None,
                              name_list=None):

        # 如果name_list为None，则初始化为一个空列表
        if name_list is None:
            name_list = []

        if params_name is None:
            return None

        # 使用deepcopy确保模型是独立的
        self.model_client.load_state_dict(OrderedDict(
            zip(params_name, local_parameters)
        ), strict=False)

        self.model_global.load_state_dict(global_parameters, strict=False)

        sim_list = []
        # 初始化用于存储展平后的logits的列表
        flattened_logits_list = []
        flattened_client_logits_list = []

        for x, y in self.trainloader:
            if len(x) <= 1:
                continue

            x = x.to(self.device)
            model_output = self.model_global(x)
            client_model_output = self.model_client(x)

            if not flattened_logits_list:
                # 展平除batch_size外的所有维度
                flattened_logits_list = [self.normalize_tensor(logit.view(logit.size(0), -1))for logit in model_output]
                flattened_client_logits_list = [self.normalize_tensor(logit.view(logit.size(0), -1)) for logit in client_model_output]
            else:
                # 在后续迭代中，累加新的batch数据
                for i, (logit, client_logit) in enumerate(zip(model_output, client_model_output)):
                    flattened_logits_list[i] = torch.cat((flattened_logits_list[i],
                                                          self.normalize_tensor(logit.view(logit.size(0), -1))))
                    flattened_client_logits_list[i] = torch.cat(
                        (flattened_client_logits_list[i],
                         self.normalize_tensor(client_logit.view(client_logit.size(0), -1))))

        # 计算每个logit与global logit之间的相似度
        for i in range(len(flattened_client_logits_list)):
            # 获取当前迭代的logit和logit_global
            logit_flat = flattened_logits_list[i]
            logit_global_flat = flattened_client_logits_list[i]

            # 计算展平后的张量之间的余弦相似度
            # sim = cosine_similarity(logit_flat.view(-1).unsqueeze(0), logit_global_flat.view(-1).unsqueeze(0))
            sim = torch.abs(torch.mean(logit_flat.unsqueeze(0) - logit_global_flat.unsqueeze(0)))
            # sim = kernel_CKA(logit_flat, logit_global_flat)
            sim_list.append(sim)

        # 计算相似度之间的差异
        delta_sim_list = [sim_list[i - 1] - sim_list[i] for i in range(1, len(sim_list))]
        print(sim_list)

        # 找到最大差异的索引
        max_delta_sim_index = torch.argmax(torch.tensor(delta_sim_list))

        layer_name = name_list[max_delta_sim_index]

        return layer_name

    def normalize_tensor(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        normalized_tensor = ((tensor - min_val) / (max_val - min_val))
        return normalized_tensor

    def gram_linear(self, x):
        """计算线性核矩阵（Gram matrix）"""
        return x @ x.T

    def centering(self, K):
        """核矩阵居中"""
        N = K.shape[0]
        unit = torch.ones(N, N, device=K.device)
        I = torch.eye(N, device=K.device)
        H = I - unit / N
        return H @ K @ H

    def hsic(self, K, L, normalize=True):
        """计算Hilbert-Schmidt独立性准则（HSIC）"""
        N = K.shape[0]
        HSH = torch.trace(K @ L)
        if normalize:
            HSH /= torch.sqrt(torch.trace(K @ K) * torch.trace(L @ L))
        return HSH / (N ** 2)

    def cka_more(self, X, Y, device='cuda'):
        """计算CKA相似度"""
        X, Y = torch.tensor(X, device=device), torch.tensor(Y, device=device)

        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)

        K = self.gram_linear(X)
        L = self.gram_linear(Y)

        K_centered = self.centering(K)
        L_centered = self.centering(L)

        return self.hsic(K_centered, L_centered, normalize=True)

    def cka(self, X, Y):
        """
        计算两个张量X和Y之间的中心核对齐（CKA）相似度。
        """
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)

        X_norm = torch.linalg.norm(X)
        Y_norm = torch.linalg.norm(Y)

        dot_product = (X.T @ Y).norm() ** 2 / (len(Y[0, :]) ** 2)
        return dot_product / (X_norm * Y_norm)

