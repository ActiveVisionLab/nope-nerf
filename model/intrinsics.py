import torch
import torch.nn as nn
import numpy as np

class LearnFocal(nn.Module):
    def __init__(self, req_grad, fx_only, order=2, init_focal=None):
        super(LearnFocal, self).__init__()

        self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
        self.order = order  # check our supplementary section.

        if self.fx_only:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal, requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
        else:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            elif isinstance(init_focal, list):
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal[0]), requires_grad=False).float()
                    coe_y = torch.tensor(np.sqrt(init_focal[1]), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal[0], requires_grad=False).float()
                    coe_y = torch.tensor(init_focal[1], requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(coe_y, requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal), requires_grad=False).float()
                    coe_y = torch.tensor(np.sqrt(init_focal), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal, requires_grad=False).float()
                    coe_y = torch.tensor(init_focal, requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(coe_y, requires_grad=req_grad)  # (1, )

    def forward(self, i=None):  # the i=None is just to enable multi-gpu training
        if self.fx_only:
            if self.order == 2:
                fxfy = torch.stack([self.fx ** 2, self.fx ** 2])
            else:
                fxfy = torch.stack([self.fx, self.fx])
        else:
            if self.order == 2:
                fxfy = torch.stack([self.fx**2, self.fy**2])
            else:
                fxfy = torch.stack([self.fx, self.fy])
        return fxfy
