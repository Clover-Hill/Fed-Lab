import copy
import time
import torch
from flcore.clients.clientdyn import clientDyn
from flcore.servers.serverbase import Server
from threading import Thread
import math

class HA_FedDyn(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientDyn)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

        # 分层聚合参数
        self.num_groups = args.num_groups if hasattr(args, 'num_groups') else 4
        self.groups = self.assign_clients_to_groups()
        self.group_models = [copy.deepcopy(args.model) for _ in range(self.num_groups)]
        self.alpha = args.alpha

        self.global_model = copy.deepcopy(args.model)
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        # 初始化每个组的服务器状态
        self.group_states = [copy.deepcopy(args.model) for _ in range(self.num_groups)]
        for group_state in self.group_states:
            for param in group_state.parameters():
                param.data = torch.zeros_like(param.data)

        # 初始化 server_state
        self.server_state = copy.deepcopy(args.model)
        for param in self.server_state.parameters():
            param.data = torch.zeros_like(param.data)

    def assign_clients_to_groups(self):
        # 简单的均等分组
        group_size = math.ceil(self.num_clients / self.num_groups)
        groups = []
        for g in range(self.num_groups):
            start = g * group_size
            end = min(start + group_size, self.num_clients)
            group = list(range(start, end))
            groups.append(group)
        return groups

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.update_server_state()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, f"Time: {self.Budget[-1]:.2f}s")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nBest local accuracy.")
        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDyn)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def add_group_parameters(self, group_idx, client_model):
        for group_param, client_param in zip(self.group_models[group_idx].parameters(), client_model.parameters()):
            group_param.data += client_param.data.clone() / len(self.groups[group_idx])

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0), "No models uploaded for aggregation."

        # 重置组模型
        for group_model in self.group_models:
            for param in group_model.parameters():
                param.data = torch.zeros_like(param.data)

        # 按组聚合客户端模型
        for client_model, client in zip(self.uploaded_models, self.selected_clients):
            client_id = client.id  # 确保 client 对象有 'id' 属性
            group_idx = self.get_group_idx(client_id)
            if group_idx == -1:
                print(f"Warning: Client {client_id} not assigned to any group.")
                continue
            self.add_group_parameters(group_idx, client_model)

        # 聚合组模型形成全局模型
        self.global_model = copy.deepcopy(self.group_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for group_model in self.group_models:
            for param, group_param in zip(self.global_model.parameters(), group_model.parameters()):
                param.data += group_param.data.clone() / self.num_groups

        # 更新 server_state
        for server_param, global_param in zip(self.server_state.parameters(), self.global_model.parameters()):
            server_param.data -= (1 / self.alpha) * global_param.data

    def get_group_idx(self, client_id):
        for g, group in enumerate(self.groups):
            if client_id in group:
                return g
        return -1  

    def update_server_state(self):
        assert (len(self.uploaded_models) > 0), "No models uploaded for updating server state."

        # 重置 server_state
        for param in self.server_state.parameters():
            param.data = torch.zeros_like(param.data)

        # 计算 server_state 的模型增量
        for client_model in self.uploaded_models:
            for server_param, client_param, global_param in zip(
                self.server_state.parameters(),
                client_model.parameters(),
                self.global_model.parameters()
            ):
                server_param.data += (client_param.data - global_param.data) / self.num_clients

        # 不进行额外的减法更新，保持 server_state 的累积