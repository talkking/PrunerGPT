from pyhessian import hessian
from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal
import torch
import numpy as np

from modelutils import to_device
import os


class Hessian:
    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True, average_by_param_size=False):
        self.hessian = hessian(model, criterion, dataloader=dataloader, cuda=cuda)
        #self.training_args = kwargs['training_args']
        param_names_dict = {}
        for name, param in self.hessian.model.named_parameters():
            param_names_dict[param] = name
        self.param_names = []
        for param in self.hessian.params:
            self.param_names.append(param_names_dict[param])
        self.average_by_param_size = average_by_param_size
        if average_by_param_size:
            self.param_num_elements = {}
            for param in self.hessian.params:
                self.param_num_elements[param_names_dict[param]] = param.numel()
        params, gradsH = get_params_grad(self.hessian.model)
        self.params = params 
        self.gradsH = gradsH  # gradient used for Hessian computation
        

    def dataloader_hv_product(self, v): 

        device = self.hessian.device
        num_data = 0  # count the number of datum points in the dataloader
        THv = [torch.zeros(p.size()).to(device) for p in self.hessian.params
              ]  # accumulate result

        #import pdb; pdb.set_trace()
        #self.hessian.model = self.hessian.model.train()
        # if self.training_args.gradient_checkpointing:
        #     self.hessian.model.gradient_checkpointing_enable()
        for inputs, targets in self.hessian.data:
            self.hessian.model.zero_grad()
            tmp_num_data = inputs.size(0)
            ####得改成逐层forward
           
            # trainer = Trainer(
            #     self.hessian.model,
            #     args=self.training_args,
            #     train_dataset=inputs,
            # )
            ###直接整个forward
            #import pdb; pdb.set_trace()
            
            #outputs = self.hessian.model(inputs.to(device))
            outputs = self.hessian.model(inputs.to(device))
            
            ### myself add begin
            outputs = outputs.logits
            outputs = outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1))
            targets = targets[:, 1:].view(-1)
            ### myself add end
            loss = self.hessian.criterion(outputs, targets.to(device))
            #self.hessian.model = self.hessian.model.to("cpu")
            #torch.cuda.empty_cache()
            #self.hessian.model = self.hessian.model.to(device)
            #import pdb; pdb.set_trace()
            #loss = loss.to("cpu")
            loss.backward(create_graph=True)
            
            params, gradsH = get_params_grad(self.hessian.model)
            
            self.hessian.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [ 
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv) 
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv 

    def trace(self, maxIter=100, tol=1e-3):
        """
        Modified from PyHessian, in order to calculate the Hessian trace for each
        parameter separately.
        """

        device = self.hessian.device
        trace_vhv = {}
        trace = 0.

        for i in range(maxIter):
            self.hessian.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            #import pdb; pdb.set_trace()
            if self.hessian.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            traces = [torch.sum(x * y).cpu().item() for (x, y) in zip(Hv, v)]
            
            for param_name, trace in zip(self.param_names, traces):
                if param_name not in trace_vhv:
                    trace_vhv[param_name] = []
                trace_vhv[param_name].append(trace)
            if abs(np.mean([trace for param_name in trace_vhv for trace in trace_vhv[param_name]]) - trace) / (trace + 1e-6) < tol:
                break
            else:
                trace = np.mean([trace for param_name in trace_vhv for trace in trace_vhv[param_name]])

        result = {}
        for param_name in trace_vhv:
            new_param_name = param_name#[6:] # for removeing the prefix "model."
            result[new_param_name] = np.abs(np.mean(trace_vhv[param_name]))
            if self.average_by_param_size:
                result[new_param_name] = result[new_param_name] / self.param_num_elements[param_name]
        #import pdb; pdb.set_trace()
        return result
        
    