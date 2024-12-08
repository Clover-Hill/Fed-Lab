import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client

class PrunedFedDynClient(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.alpha = args.alpha
        self.pruning_percentage = args.pruning_percentage if hasattr(args, 'pruning_percentage') else 0.2

        self.global_model_vector = None
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    def prune_model(self):
        # 全局剪枝基于权重的绝对值
        all_weights = torch.cat([param.view(-1) for param in self.model.parameters()])
        threshold = torch.quantile(all_weights.abs(), self.pruning_percentage)
        with torch.no_grad():
            for param in self.model.parameters():
                param.data = torch.where(param.data.abs() < threshold, torch.tensor(0.0).to(param.device), param.data)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                if self.global_model_vector is not None:
                    v1 = model_parameter_vector(self.model)
                    loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.global_model_vector is not None:
            v1 = model_parameter_vector(self.model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # 模型剪枝
        self.prune_model()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model_vector = model_parameter_vector(model).detach().clone()

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                if self.global_model_vector is not None:
                    v1 = model_parameter_vector(self.model)
                    loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)
