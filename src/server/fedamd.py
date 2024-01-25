from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from collections import Counter
import torch
from rich.progress import track

from fedavg_V1 import FedAvgServer, get_fedavg_argparser
from src.config.utils import trainable_params

import re
from collections import defaultdict

import os


def get_pfedsim_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("-wr", "--warmup_round", type=float, default=0.1)
    parser.add_argument("--server_momentum", type=float, default=0.9)
    return parser


class FedSoftPerServer(FedAvgServer):
    def __init__(
            self,
            algo: str = "FedAMD",
            args: Namespace = None,
            unique_model=False,
            default_trainer=True,
    ):
        if args is None:
            args = get_pfedsim_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.test_flag = False
        self.weight_matrix = torch.eye(self.client_num, device=self.device)
        self.startup_round = 0
        self.layer_gamma = {}
        self.two_smallest_elements = {}
        self.phi = 20
        self.weight_norm_list = []
        self.global_optimizer = torch.optim.SGD(
            list(self.global_params_dict.values()),
            lr=1.0,
            momentum=self.args.server_momentum,
            nesterov=True,
        )


    def train(self):
        self.unique_model = True
        fedsper_progress_bar = track(
            range(self.startup_round, self.args.global_epoch),
            "[bold green]Personalizing...",
            console=self.logger.stdout,
        )

        self.client_trainable_params = [
            trainable_params(self.global_params_dict, detach=True)
            for _ in self.train_clients
        ]
        all_layer = {}
        for E in fedsper_progress_bar:
            self.trainer.personal_params_name = [
                name for name in self.model.state_dict() if any(ele in name for ele in self.two_smallest_elements)
            ]

            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            self.selected_clients = self.client_sample_stream[E]

            client_params_cache = []
            delta_cache = []
            weight_cache = []

            for client_id in self.selected_clients:
                client_pers_params = self.generate_client_params(client_id)
                (
                    client_params,
                    delta,
                    weight,
                    self.client_stats[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    new_parameters=client_pers_params,
                    return_diff=True,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )
                client_params_cache.append(client_params)
                delta_cache.append(delta)
                weight_cache.append(weight)

            if self.current_epoch < self.phi:
                self.find_most_common_layer()
                all_layer[self.two_smallest_elements[0]] = all_layer.setdefault(self.two_smallest_elements[0], 0) + 1
                print(f"{self.current_epoch}: Two smallest elements based on 'weight':", self.two_smallest_elements)
            if self.current_epoch == self.phi - 1:
                max_key = max(all_layer, key=all_layer.get)
                self.two_smallest_elements = [max_key]
                print(f": What I ultimately chose is':", self.two_smallest_elements)

            self.weight_norm_list = [weight / sum(weight_cache) for weight in weight_cache]
            self.update_client_params(client_params_cache)
            self.update_weight_matrix(delta_cache)
            self.log_info()

    def find_most_common_layer(self):
        layer_counts = defaultdict(int)
        client_layer = {}
        # 打开并读取日志文件的内容
        with open(f'kl_log_{self.args.algo}.txt', 'r') as file:
            lines = file.readlines()
            content = ''.join(lines)
            for client_id in self.selected_clients:
                pattern = rf'{client_id} Layer with minimum contribution to output y: (\w+)'
                matches = re.findall(pattern, content)
                for match in matches:
                    layer_counts[match] += 1
                    most_common_layer = max(layer_counts, key=layer_counts.get)
                    client_layer[client_id] = most_common_layer
            value_counts = Counter(client_layer.values())

            most_common_layer = max(value_counts, key=value_counts.get)
            self.two_smallest_elements = [most_common_layer]

        os.remove(f'kl_log_{self.args.algo}.txt')

    @torch.no_grad()
    def generate_client_params(self, client_id):
        new_parameters = OrderedDict(
            zip(
                self.trainable_params_name,
                deepcopy(self.client_trainable_params[client_id]),
            )
        )

        if self.current_epoch < self.phi:
            if sum(self.weight_matrix[client_id]) > 1:
                weights = self.weight_matrix[client_id].clone()
                weights = torch.exp(weights)
                weights /= weights.sum()
                N_dim = weights.shape[0]
                uniform_matrix = torch.full((N_dim,), 1 / N_dim).cuda()

                for name, layer_params in zip(
                        self.trainable_params_name, zip(*self.client_trainable_params)
                ):
                    new_parameters[name] = torch.sum(
                        torch.stack(layer_params, dim=-1) * uniform_matrix, dim=-1
                    )
        else:
            if not self.test_flag:
                layer_indices = [i for i, param_name in enumerate(self.trainable_params_name)
                                 if param_name in self.trainer.personal_params_name]
                max_index = max(layer_indices) if layer_indices else None
                min_index = min(layer_indices) if layer_indices else None
                if sum(self.weight_matrix[client_id]) > 1:
                    weights = self.weight_matrix[client_id].clone()
                    weights = torch.exp(weights)
                    weights /= weights.sum()
                    for i, (name, layer_params) in enumerate(zip(
                            self.trainable_params_name, zip(*self.client_trainable_params)
                    )):
                        if i < min_index:
                            layer_params = [a * b for a, b in zip(layer_params, self.weight_norm_list)]
                            new_parameters[name] = torch.sum(
                                torch.stack(layer_params, dim=-1), dim=-1)
                        elif i > max_index:
                            new_parameters[name] = torch.sum(
                                torch.stack(layer_params, dim=-1) * weights, dim=-1
                            )
                        else:
                            new_parameters[name] = deepcopy(self.global_params_dict[name])

        return new_parameters


    @torch.no_grad()
    def update_weight_matrix(self, delta_cache):
        for idx_i, i in enumerate(self.selected_clients):
            client_params_i = delta_cache[idx_i]

            for idx_j, j in enumerate(self.selected_clients[idx_i + 1:]):
                client_params_j = delta_cache[idx_i + idx_j + 1]
                # 将张量从GPU移到CPU
                if 'classifier' in self.two_smallest_elements:
                    classifier_weight_i_cpu = client_params_i["classifier.weight"].cpu()
                    classifier_weight_j_cpu = client_params_j["classifier.weight"].cpu()
                else:
                    classifier_weight_i_cpu = client_params_i[str(self.two_smallest_elements[0]) + ".1.weight"].cpu()
                    classifier_weight_j_cpu = client_params_j[str(self.two_smallest_elements[0]) + ".1.weight"].cpu()

                sim_ij_classifier = max(
                    0,
                    torch.cosine_similarity(
                        classifier_weight_i_cpu,
                        classifier_weight_j_cpu,
                        dim=-1,
                    ).mean().item()
                )
                sim_ij_classifier = torch.tensor(sim_ij_classifier).to('cpu')

                self.weight_matrix[j, i] = self.weight_matrix[j, i] = sim_ij_classifier


if __name__ == "__main__":
    server = FedSoftPerServer()
    server.run()
