import torch
from torch import nn


class MultiOutputLayers(nn.Module):
    """NN for MTL with hard parameter sharing"""

    def __init__(
        self,
        out_layer,
        output_nums,
    ):
        super().__init__()

        self.task_nets = nn.ModuleList()
        for _ in range(output_nums):
            self.task_nets.append(out_layer)

    def forward(self, x):
        res = []

        for i in range(self.output_nums):
            lm = self.task_nets[i](x[i, ...])
            res.append(lm)

        res = torch.stack(res, dim=0)
        return res


def split_dataset(dataset, split_num, tokenizer):
    text = dataset["text"]
    sep_token = tokenizer.sep_token
    eos_token = tokenizer.eos_token

    res = []

    for t in text:
        t = t.split(eos_token)[0]
        t = t.split(sep_token)
        header = sep_token.join(t[:-split_num])

        new_t = []
        for i in range(split_num):
            index = -split_num + i
            new_t.append(header + sep_token + t[index] + eos_token)

        res.append(new_t)

    dataset["text"] = res

    return dataset
