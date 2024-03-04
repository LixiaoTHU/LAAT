from enum import Enum
import torch
import numpy as np


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class SimpleMeter(object):
    """Always stores the current value"""

    def __init__(self, name="", fmt="{}"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        return self

    def update(self, val):
        self.val = val
        return self

    def result(self):
        return self.val

    def __str__(self):
        fmtstr = "{name}=" + self.fmt.replace("{", "{val")
        return fmtstr.format(**self.__dict__)

    def summary(self):
        return self.__str__()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt="{}", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self

    def result(self):
        if self.summary_type is Summary.NONE:
            val = self.val
        elif self.summary_type is Summary.AVERAGE:
            val = self.avg
        elif self.summary_type is Summary.SUM:
            val = self.sum
        elif self.summary_type is Summary.COUNT:
            val = self.count
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)
        return val

    def __str__(self):
        fmtstr = (
            "{name}="
            + self.fmt.replace("{", "{val")
            + " ("
            + self.fmt.replace("{", "{avg")
            + ")"
        )
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name}=" + self.fmt.replace("{", "{avg")
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name}=" + self.fmt.replace("{", "{sum")
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name}=" + self.fmt.replace("{", "{count")
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class DistributionMeter(object):
    """Track all values to compute distribution parameters."""

    def __init__(self, name="", fmt="{}"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.all_val = []
        return self

    def update(self, val):
        self.val = val
        self.all_val.append(val)
        return self

    def collect(self):
        self.mean = np.mean(self.all_val)
        if len(self.all_val) > 1:
            self.std = np.std(self.all_val, ddof=1)
        else:
            self.std = 0

    def result(self):
        self.collect()
        return self.mean

    def __str__(self):
        self.collect()
        fmtstr = (
            "{name}="
            + self.fmt.replace("{", "{mean")
            + "±"
            + self.fmt.replace("{", "{std")
        )
        return fmtstr.format(**self.__dict__)

    def summary(self):
        return self.__str__()


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_results(meters):
    return dict([(m.name, m.result()) for m in meters])


def get_summary(meters, format="{name}={val}", delim="  "):
    r"""Get summary string from meters.

    Suggested formats:
      - format="{name}={val}", delim="  "
      - format="{val}", delim="\t" (note: use format="{name}", delim="\t" to get header)
      - format="csv" (note: delim is ignored and set to "," in csv format)
    """
    if format == "csv":
        return ",".join(m.result() for m in meters)
    l = []
    for m in meters:
        l.append(
            format.replace("{name}", m.name).replace("{val}", m.fmt.format(m.result()))
        )
    return delim.join(l)
