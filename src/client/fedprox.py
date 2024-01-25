from fedavg import FedAvgClient
from src.config.utils import trainable_params
import torch
import torch.nn as nn


device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class FedProxClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super(FedProxClient, self).__init__(model, args, logger, device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def leave_one_out_control_variates(self, losses):
        total_loss = sum(losses)
        loo_losses = [(total_loss - loss) / (len(losses) - 1) for loss in losses]

        # Convert the list of tensor losses to a single tensor
        return torch.stack(loo_losses)

    def fit(self):
        self.model.train()
        global_params = trainable_params(self.model, detach=True)
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                if self.model.name == "DecoupledModel":
                    logit = self.model(x)
                else:
                    logit_vae_list = self.model(x)
                    logit = logit_vae_list[-1]
                loss = self.criterion(logit, y)

                self.optimizer.zero_grad()
                loss.backward()
                for w, w_t in zip(trainable_params(self.model), global_params):
                    w.grad.data += self.args.mu * (w.data - w_t.data)
                self.optimizer.step()
