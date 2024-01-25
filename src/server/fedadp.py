from argparse import ArgumentParser, Namespace
from copy import deepcopy
from collections import OrderedDict, Counter

import torch
from rich.progress import track

from fedavg_V1 import FedAvgServer, get_fedavg_argparser
from src.config.utils import trainable_params
from src.client.fedadp import FedADPlient


def get_fedadp_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("-wr", "--warmup_round", type=float, default=0)
    parser.add_argument("--server_momentum", type=float, default=0.9)
    return parser



class FedADPServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedADP",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        if args is None:
            args = get_fedadp_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedADPlient(deepcopy(self.model), self.args, self.logger, self.device)
        self.voting_array = torch.zeros(self.client_num, device=self.device)
        self.two_smallest_elements = {}
        self.weight_matrix = torch.eye(self.client_num, device=self.device)
        self.client_layers = {}
        self.global_optimizer = torch.optim.SGD(
            list(self.global_params_dict.values()),
            lr=1.0,
            momentum=self.args.server_momentum,
            nesterov=True,
        )

    def train(self):

        # Personalization Phase
        self.unique_model = True
        pfedsim_progress_bar = track(
            range(self.args.global_epoch),
            "[bold green]Personalizing...",
            console=self.logger.stdout,
        )

        self.client_trainable_params = [
            trainable_params(self.global_params_dict, detach=True)
            for _ in self.train_clients
        ]

        name_list = ['fc1', 'fc2', 'fc3', 'fc4', 'classifier']

        for E in pfedsim_progress_bar:

            self.two_smallest_elements = ["fc2"]

            self.trainer.personal_params_name = [
                name for name in self.model.state_dict() if any(ele in name for ele in self.two_smallest_elements)
            ]

            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            client_params_cache = []
            weight_params_cache = []
            client_pers_params_cache = []

            for client_id in self.selected_clients:
                if client_id in self.client_layers:
                    name = [name for name in self.trainable_params_name if self.client_layers[client_id] in name]
                else:
                    self.client_layers[client_id] = ''
                    name = []

                client_pers_params = self.generate_client_params(client_id)
                (
                    client_params,
                    delta,
                    _,
                    self.client_stats[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    new_parameters=client_pers_params,
                    return_diff=True,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                    name=name,  # 添加新参数
                )

                client_params_cache.append(deepcopy(client_params))

                # client_pers_params = self.generate_client_params(client_id)
                # (
                #     client_params,
                #     weight_params,
                #     self.client_stats[client_id][E],
                # ) = self.trainer.train(
                #     client_id=client_id,
                #     local_epoch=self.clients_local_epoch[client_id],
                #     new_parameters=client_pers_params,
                #     return_diff=False,
                #     verbose=((E + 1) % self.args.verbose_gap) == 0,
                # )

            self.update_client_params(client_params_cache)

            self.log_info()

            layer_name = []

            for index, _ in enumerate(self.selected_clients):
                client_pers_params = client_params_cache[index]

                layer_name.append(self.trainer.voting_personal_layer(
                    local_parameters=client_pers_params,
                    global_parameters=self.global_params_dict,
                    name_list=name_list,
                    params_name=self.trainable_params_name,
                ))

            # 使用Counter来计算每个元素的出现次数
            layer_name_counter = Counter(layer_name)
            print(layer_name)

            # 找到出现次数最多的元素
            most_common_result = layer_name_counter.most_common(1)[0][0]
            print("选层结果:\t" + most_common_result)
            for index, client_id in enumerate(self.selected_clients):
                self.client_layers[client_id] = most_common_result


    @torch.no_grad()
    def generate_client_params(self, client_id):
        new_parameters = OrderedDict(
            zip(
                self.trainable_params_name,
                deepcopy(self.client_trainable_params[client_id]),
            )
        )
        if not self.test_flag:
                # 获取N维方阵的维度
                N_dim = 100

                # 在CUDA上生成一个N x N的方阵，每个元素值为1/N
                uniform_matrix = torch.full((N_dim,), 1 / N_dim).cuda()

                for name, layer_params in zip(
                        self.trainable_params_name, zip(*self.client_trainable_params)
                ):

                    new_parameters[name] = torch.sum(
                        torch.stack(layer_params, dim=-1) * uniform_matrix, dim=-1
                    )

        return new_parameters


if __name__ == "__main__":
    server = FedADPServer()
    server.run()