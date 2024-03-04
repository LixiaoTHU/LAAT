import os, argparse, time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import clip

from models import *
from datasets import *
from attacks import *
import utils
import awp


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Adversarial Training for CLIP.")
    parser.add_argument("--data_dir", type=str, default="~/data1/dataset")
    parser.add_argument("--exp_name", type=str, default="~/data1/output/clip_adv_zeroshot")
    parser.add_argument("--load", type=int)
    parser.add_argument("--load_best", action="store_true", default=False)
    parser.add_argument("--load_pretrained", type=str)
    parser.add_argument("--ckpt_interval", type=int, default=20)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument(
        "--train_type", default="AT", choices=["AT", "TRADES", "TRADES-cos"]
    )
    # Always use 'metric' for few-shot
    parser.add_argument(
        "--dataset",
        default="CIFAR100FS",
        choices=["CIFAR100FS", "miniImageNet"],
    )
    parser.add_argument("--epsilon", default=8, type=int, help="perturbation")
    parser.add_argument("--bs", default=64 * 2, type=int, help="batch size")
    parser.add_argument("--lr", default=0.05, type=float, help="lr")
    parser.add_argument(
        "--modelname", default="ViT-B/16", choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"]
    )
    parser.add_argument(
        "--model",
        default="ResNet12",
        choices=["ResNet12", "Conv4-512"],
        help="vision model used",
    )
    parser.add_argument(
        "--head",
        default="cos-clip",
        choices=["cos-clip", "cos-span", "linear"],
        help="classification head",
    )
    parser.add_argument(
        "--loss",
        default="cos-ce",
        choices=["cos-ce", "cos", "acos", "l2"],
        help="loss type",
    )
    parser.add_argument("--use_linear", action="store_true", default=False)
    parser.add_argument("--all_epoch", default=100, type=int, help="epoch")
    # Possible tau=0.07
    parser.add_argument("--tau", default=1.0, type=float, help="temporature")
    parser.add_argument("--beta", default=1.0, type=float, help="for TRADES")
    parser.add_argument("--drop_rate", default=0.0, type=float, help="dropout")
    parser.add_argument("--attack", default="PGD", choices=["PGD", "FGSM", "CW", "AA", "none"])
    parser.add_argument("--suffix", type=str, help="checkpoint suffix")
    # AWP settings, possible gamma=5e-3
    parser.add_argument("--awp_gamma", default=0.0, type=float)
    parser.add_argument("--awp_warmup", default=0, type=int)

    # n-shot setting
    parser.add_argument(
        "--n_class", default=5, type=int, help="number of classes in n-shot"
    )
    parser.add_argument(
        "--n_support",
        default=5,
        type=int,
        help="number of support samples each class in n-shot",
    )
    parser.add_argument(
        "--n_query", default=20, type=int, help="number of queries each class in n-shot"
    )
    parser.add_argument(
        "--n_val", default=100, type=int, help="number of validation repeats"
    )
    parser.add_argument(
        "--n_test", default=2000, type=int, help="number of test repeats"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="This is a photo of a {}",
        help="prompt to embed class label",
    )
    # Possible text weight 2
    parser.add_argument(
        "--text_weight",
        type=float,
        default=0.0,
        help="text feature weight when constructing metric",
    )
    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args):
    if epoch >= 0.6 * args.all_epoch:
        lr = args.lr * 0.1
    elif epoch >= 0.8 * args.all_epoch:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train(epoch, model, optimizer, adversary, awp_adversary, trainloader, device, args):
    start = time.time()
    if args.loss == "cos-ce":
        base_criterion = nn.CrossEntropyLoss()
    elif args.loss == "cos" or args.loss == "acos" or args.loss == "l2":
        base_criterion = utils.DotLoss()
    else:
        raise ValueError(args.loss)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    epoch_meter = utils.SimpleMeter("epoch").update(epoch)
    lr = utils.SimpleMeter("lr", fmt="{:.4f}")
    train_time = utils.SimpleMeter("train_time", fmt="{:.1f}")
    train_clean_loss = utils.AverageMeter("train_clean_loss", fmt="{:.4f}")
    train_clean_acc = utils.AverageMeter("train_clean_acc", fmt="{:.2f}%")
    train_robust_loss = utils.AverageMeter("train_robust_loss", fmt="{:.4f}")
    train_robust_acc = utils.AverageMeter("train_robust_acc", fmt="{:.2f}%")
    meters = [
        epoch_meter,
        lr,
        train_time,
        train_clean_loss,
        train_clean_acc,
        train_robust_loss,
        train_robust_acc,
    ]
    lr.update(adjust_learning_rate(optimizer, epoch, args))
    for batch_idx, (X, y) in enumerate(trainloader):
        X, y = X.to(device), y.to(device)
        X_adv = adversary.attack(model, X, y)
        # calculate adversarial weight perturbation and perturb it
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=X_adv, targets=y)
            awp_adversary.perturb(awp)
        model.train()
        optimizer.zero_grad()
        if args.train_type == "AT":
            logits_adv = model(X_adv)
            loss_adv = base_criterion(logits_adv, y)
            loss = loss_adv
        elif args.train_type == "TRADES":
            logits_clean = model(X)
            logits_adv = model(X_adv)
            logits_clean_2 = model(X)
            loss_clean = base_criterion(logits_clean, y)
            loss_kl = criterion_kl(
                F.log_softmax(logits_adv, dim=1), F.softmax(logits_clean_2, dim=1)
            )
            loss = loss_clean + args.beta * loss_kl
            loss_adv = loss_kl
        elif args.train_type == "TRADES-cos":
            logits_adv = model(X_adv)
            feat_clean = model[:-1](X)
            feat_adv = model[:-1](X_adv)
            logits_clean = model[-1](feat_clean)
            loss_adv = base_criterion(logits_adv, y)
            feat_clean = feat_clean / feat_clean.norm(dim=-1, keepdim=True)
            feat_adv = feat_adv / feat_adv.norm(dim=-1, keepdim=True)
            loss_kl = -torch.sum(feat_clean * feat_adv, dim=-1).mean()
            loss = loss_adv + args.beta * loss_kl
            loss_clean = loss_kl
        else:
            raise ValueError(args.train_type)
        loss.backward()
        optimizer.step()
        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)
        model.eval()
        with torch.no_grad():
            if args.train_type == "AT":
                logits_clean = model(X)
                loss_clean = base_criterion(logits_clean, y)
        batch_size = X.shape[0]
        clean_acc = logits_clean.argmax(dim=1).eq(y).sum() / batch_size
        robust_acc = logits_adv.argmax(dim=1).eq(y).sum() / batch_size
        train_clean_loss.update(loss_clean.item(), batch_size)
        train_clean_acc.update(clean_acc.item() * 100, batch_size)
        train_robust_loss.update(loss_adv.item(), batch_size)
        train_robust_acc.update(robust_acc.item() * 100, batch_size)
    end = time.time()
    train_time.update(end - start)
    return meters


@torch.no_grad()
def test(is_val, model, test_text_features, adversary, testloader, device, args):
    start = time.time()
    if args.loss == "cos-ce":
        base_criterion = nn.CrossEntropyLoss()
    elif args.loss == "cos" or args.loss == "acos" or args.loss == "l2":
        base_criterion = utils.DotLoss()
    else:
        raise ValueError(args.loss)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    if is_val:
        prefix = "val"
        repeats = args.n_val
        test_clean_acc = utils.AverageMeter(prefix + "_clean_acc", fmt="{:.2f}%")
        test_robust_acc = utils.AverageMeter(prefix + "_robust_acc", fmt="{:.2f}%")
    else:
        prefix = "test"
        repeats = args.n_test
        test_clean_acc = utils.SamplesMeter(prefix + "_clean_acc", fmt="{:.2f}")
        test_robust_acc = utils.SamplesMeter(prefix + "_robust_acc", fmt="{:.2f}")
    test_time = utils.SimpleMeter(prefix + "_time", fmt="{:.1f}")
    test_clean_loss = utils.AverageMeter(prefix + "_clean_loss", fmt="{:.4f}")
    test_robust_loss = utils.AverageMeter(prefix + "_robust_loss", fmt="{:.4f}")
    meters = [
        test_time,
        test_clean_loss,
        test_clean_acc,
        test_robust_loss,
        test_robust_acc,
    ]
    model.eval()
    p = args.n_class * args.n_support
    for _ in range(repeats):
        local_test_clean_acc = utils.AverageMeter()
        local_test_robust_acc = utils.AverageMeter()
        for batch_idx, (X, y) in enumerate(testloader):
            text_features = test_text_features[testloader.sampler.classes]
            if args.head != "linear":
                model[-1].gts.copy_(text_features)
            y = testloader.sampler.convert_target(y)
            X, y = X.to(device), y.to(device)
            feat_clean = model[:-1](X)
            if batch_idx == 0 and p > 0:
                # of support set
                feat_s = feat_clean[:p]
                feat_s = feat_s.reshape(args.n_support, args.n_class, -1).mean(dim=0)
                if args.text_weight > 0:
                    feat_s = args.text_weight * text_features + args.n_support * feat_s
                if args.loss != "l2":
                    feat_s = feat_s / feat_s.norm(dim=-1, keepdim=True)
                model[-1].gts.copy_(feat_s.detach())
                # of query set
                X = X[p:]
                y = y[p:]
                feat_clean = feat_clean[p:]
            logits_clean = model[-1](feat_clean)
            if args.attack == "AA":
                X_adv = adversary.run_standard_evaluation(X, y, bs=args.bs)
            else:
                X_adv = adversary.attack(model, X, y)
            if args.train_type == "AT":
                logits_adv = model(X_adv)
                loss_adv = base_criterion(logits_adv, y)
                loss_clean = base_criterion(logits_clean, y)
            elif args.train_type == "TRADES":
                logits_adv = model(X_adv)
                loss_clean = base_criterion(logits_clean, y)
                loss_kl = criterion_kl(
                    F.log_softmax(logits_adv, dim=1), F.softmax(logits_clean, dim=1)
                )
                # loss = loss_clean + args.beta * loss_kl
                loss_adv = loss_kl
            elif args.train_type == "TRADES-cos":
                feat_adv = model[:-1](X_adv)
                logits_adv = model[-1](feat_adv)
                loss_clean = base_criterion(logits_adv, y)
                feat_clean = feat_clean / feat_clean.norm(dim=-1, keepdim=True)
                feat_adv = feat_adv / feat_adv.norm(dim=-1, keepdim=True)
                loss_kl = -torch.sum(feat_clean * feat_adv, dim=-1).mean()
                # loss = loss_clean + args.beta * loss_kl
                loss_adv = loss_kl
            else:
                raise ValueError(args.train_type)
            batch_size = X.shape[0]
            clean_acc = logits_clean.argmax(dim=1).eq(y).sum() / batch_size
            robust_acc = logits_adv.argmax(dim=1).eq(y).sum() / batch_size
            test_clean_loss.update(loss_clean.item(), batch_size)
            local_test_clean_acc.update(clean_acc.item() * 100, batch_size)
            test_robust_loss.update(loss_adv.item(), batch_size)
            local_test_robust_acc.update(robust_acc.item() * 100, batch_size)
        test_clean_acc.update(local_test_clean_acc.avg, local_test_clean_acc.count)
        test_robust_acc.update(local_test_robust_acc.avg, local_test_robust_acc.count)
    end = time.time()
    test_time.update(end - start)
    return meters


def get_exp_path(args):
    # Expand user for all directories
    args.data_dir = os.path.expanduser(args.data_dir)
    args.exp_name = os.path.expanduser(args.exp_name)
    savemodelname = args.modelname.replace("/", "_")
    few_shot_name = f"{args.n_class}way_{args.n_support}shot_metric"
    model_args_name = f"tau{args.tau}"
    train_name = args.train_type
    if args.awp_gamma > 0.0:
        train_name += f"-AWP{args.awp_gamma}"
        if args.awp_warmup > 0:
            train_name += f"-warmup{args.awp_warmup}"
    exp_name_list = [
        args.dataset,
        "CLIP_" + savemodelname,
        f"{args.model}_{args.head}",
        model_args_name,
        train_name,
        few_shot_name,
    ]
    exp_path = os.path.join(args.exp_name, *exp_name_list)
    # Append run index or suffix
    if args.suffix is None:
        run_idx = 1
        while os.path.exists(os.path.join(exp_path, str(run_idx))):
            run_idx += 1
        exp_path = os.path.join(exp_path, str(run_idx))
    else:
        exp_path = os.path.join(exp_path, args.suffix)
    return exp_path


def main():
    # print(clip.available_models())

    args = get_arguments()
    exp_path = get_exp_path(args)
    print("Exp path is", exp_path)
    if args.eval:
        assert os.path.exists(exp_path)
    else:
        os.makedirs(exp_path, exist_ok=True)
    log_file_path = os.path.join(exp_path, "eval.log" if args.eval else "train.log")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    eps = args.epsilon / 255
    pgd_alpha = eps * 2 / 8
    if args.awp_gamma <= 0.0:
        args.awp_warmup = torch.inf

    device = utils.get_device()
    # device_list = [
    #     torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())
    # ]
    # print(device_list)

    logger = utils.setup_logger(name="CLIP_" + args.modelname, log_file=log_file_path)
    logger.info(args)

    # Import dataset
    if args.dataset == "CIFAR100FS":
        input_res = 32
        train_preprocess = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ]
        )
        test_preprocess = transforms.ToTensor()
        trainset = CIFAR100FS(args.data_dir, "train", transform=train_preprocess)
        valset = CIFAR100FS(args.data_dir, "val", transform=test_preprocess)
        testset = CIFAR100FS(args.data_dir, "test", transform=test_preprocess)
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    elif args.dataset == "miniImageNet":
        input_res = 80
        train_preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.RandomCrop(80, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.CenterCrop(80),
                transforms.ToTensor(),
            ]
        )
        trainset = MiniImageNet(args.data_dir, "train", transform=train_preprocess)
        valset = MiniImageNet(args.data_dir, "val", transform=test_preprocess)
        testset = MiniImageNet(args.data_dir, "test", transform=test_preprocess)
        mean = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
    else:
        raise ValueError(args.dataset)
    trainloader = data.DataLoader(
        dataset=trainset,
        batch_size=args.bs,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    valloader = data.DataLoader(
        dataset=testset,
        sampler=NShotTaskSampler(
            testset.ids,
            args.n_class,
            args.n_support,
            args.n_query,
            num_batches=args.n_val,
        ),
        batch_size=65536,
        pin_memory=True,
        num_workers=2,
    )
    testloader = data.DataLoader(
        dataset=testset,
        sampler=NShotTaskSampler(
            testset.ids, args.n_class, args.n_support, args.n_query
        ),
        batch_size=65536,
        pin_memory=True,
        num_workers=2,
    )
    mean = torch.tensor(mean, device=device).reshape(3, 1, 1)
    std = torch.tensor(std, device=device).reshape(3, 1, 1)

    # models
    clip_model, clip_preprocess = clip.load(args.modelname, device=device)
    clip_model.eval()

    train_labels = list(
        map(lambda _: args.prompt.format(_.replace("_", " ")), trainset.classlist)
    )
    train_text_tokens = clip.tokenize(train_labels).to(device)
    train_text_features = clip_model.encode_text(train_text_tokens).float().detach()
    test_labels = list(
        map(lambda _: args.prompt.format(_.replace("_", " ")), testset.classlist)
    )
    test_text_tokens = clip.tokenize(test_labels).to(device)
    test_text_features = clip_model.encode_text(test_text_tokens).float().detach()
    if args.head == "cos-span":
        import numpy as np

        if args.dataset == "CIFAR100FS":
            feature_file = "anchors/cifarfs_clip_weight_a.npy"
            class_file = "anchors/cifarfs_classes.txt"
        elif args.dataset == "miniImageNet":
            feature_file = "anchors/miniimagenet_clip_weight_a.npy"
            class_file = "anchors/miniimagenet_classes.txt"
        else:
            raise ValueError(args.dataset)
        # Verify classes
        all_labels = trainset.classlist + valset.classlist + testset.classlist
        with open(class_file, "r") as f:
            for idx, line in enumerate(f):
                expected_class = all_labels[idx].replace("_", " ")
                real_class = line.rstrip().replace("_", " ")
                assert (
                    real_class == expected_class
                ), f"Class {idx} '{real_class}' different from '{expected_class}'"
        all_features = torch.from_numpy(np.load(feature_file)).float()
        train_text_features = all_features[: len(trainset.classlist)]
        test_text_features = all_features[-len(testset.classlist) :]

    if args.loss != "l2":
        train_text_features = train_text_features / train_text_features.norm(
            dim=-1, keepdim=True
        )
        test_text_features = test_text_features / test_text_features.norm(
            dim=-1, keepdim=True
        )

    if args.model == "ResNet12":
        if args.dataset == "miniImageNet":
            backbone = resnet12(
                avg_pool=False, drop_rate=args.drop_rate, dropblock_size=5
            )
        else:
            backbone = resnet12(
                avg_pool=False, drop_rate=args.drop_rate, dropblock_size=2
            )
        feature_dim = backbone.out_dim
    elif args.model == "Conv4-512":
        # backbone = conv4_512()
        # feature_dim = backbone.channels * (input_res // 8) ** 2
        backbone = ConvNet(4, 512, postprocess="avgpool")
        feature_dim = backbone.channels
    else:
        raise ValueError(args.model)
    if args.use_linear:
        backbone = nn.Sequential(
            backbone, nn.Linear(feature_dim, train_text_features.shape[-1], False)
        )
        feature_dim = train_text_features.shape[-1]

    if args.head == "cos-clip" or args.head == "cos-span":
        if args.loss == "l2":
            train_head = L2Similarity(train_text_features)
            test_head = L2Similarity(torch.empty((args.n_class, feature_dim)))
        else:
            train_head = CosineSimilarity(
                train_text_features, args.tau, use_acos=(args.loss == "acos")
            )
            test_head = CosineSimilarity(
                torch.empty((args.n_class, feature_dim)),
                args.tau,
                use_acos=(args.loss == "acos"),
            )
    elif args.head == "linear":
        train_head = nn.Linear(feature_dim, len(trainset.classlist), False)
        test_head = CosineSimilarity(torch.empty((args.n_class, feature_dim)))
    else:
        raise ValueError(args.head)

    train_model = nn.Sequential(Normalize(mean, std), backbone, train_head)
    test_model = nn.Sequential(*train_model[:-1], test_head)
    train_model = train_model.to(device)
    test_model = test_model.to(device)

    # adversary
    if args.train_type == "TRADES":
        train_adversary = PGDTrades(eps, pgd_alpha, 7)
    else:
        train_adversary = PGD(eps, pgd_alpha, 7)
    if args.attack == "PGD":
        test_adversary = PGD(eps, pgd_alpha, 20)
    elif args.attack == "FGSM":
        test_adversary = FGSM(eps)
    elif args.attack == "CW":
        test_adversary = CWLinf(eps)
    elif args.attack == "AA":
        from autoattack import AutoAttack
        aa_log_path = os.path.join(exp_path, "autoattack.log")
        test_adversary = AutoAttack(test_model, eps=eps, log_path=aa_log_path)
    else:
        test_adversary = None

    # AWP
    if args.loss == "cos-ce":
        base_criterion = nn.CrossEntropyLoss()
    elif args.loss == "cos" or args.loss == "acos" or args.loss == "l2":
        base_criterion = utils.DotLoss()
    else:
        raise ValueError(args.loss)
    proxy = copy.deepcopy(train_model)
    proxy_optim = torch.optim.SGD(proxy.parameters(), lr=0.01)
    awp_adversary = awp.AdvWeightPerturb(
        train_model, base_criterion, proxy, proxy_optim, args.awp_gamma
    )

    # optimizer
    optimizer = torch.optim.SGD(
        backbone.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9
    )

    # load checkpoint
    ENV = {"global_step": 0, "best_clean_acc": 0.0, "best_robust_acc": 0.0}
    starting_epoch = 0
    checkpoint = None
    if args.load is not None:
        checkpoint = utils.load_model(
            os.path.join(exp_path, f"model_{args.load}"), backbone, optimizer
        )
        logger.info("Load epoch {}".format(checkpoint["epoch"]))
    if args.load_best:
        checkpoint = utils.load_model(
            os.path.join(exp_path, "model_best"), backbone, optimizer
        )
        logger.info("Load best epoch {}".format(checkpoint["epoch"]))
    if args.load_pretrained is not None:
        checkpoint = utils.load_model(args.load_pretrained, backbone, optimizer)
    if checkpoint is not None:
        if "epoch" in checkpoint:
            starting_epoch = checkpoint["epoch"]
        for k in ENV.keys():
            if k in checkpoint:
                ENV[k] = checkpoint[k]

    if not args.eval:
        for epoch in range(starting_epoch, args.all_epoch):
            train_meters = train(
                epoch,
                train_model,
                optimizer,
                train_adversary,
                awp_adversary,
                trainloader,
                device,
                args,
            )
            train_results = utils.get_results(train_meters)
            val_meters = test(
                True,
                test_model,
                test_text_features,
                test_adversary,
                valloader,
                device,
                args,
            )
            val_results = utils.get_results(val_meters)

            is_best = val_results["val_robust_acc"] > ENV["best_robust_acc"]
            if is_best:
                ENV["best_robust_acc"] = val_results["val_robust_acc"]
                ENV["best_clean_acc"] = val_results["val_clean_acc"]

            # save checkpoint
            if (epoch + 1) % args.ckpt_interval == 0 or is_best:
                state = {
                    "epoch": epoch,
                    "model_state_dict": backbone.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                for k, v in val_results.items():
                    if k.endswith("loss") or k.endswith("acc"):
                        state[k] = v
                for k, v in train_results.items():
                    if k.endswith("loss") or k.endswith("acc"):
                        state[k] = v
            if (epoch + 1) % args.ckpt_interval == 0:
                with open(os.path.join(exp_path, f"model_{epoch}.pt"), "wb") as f:
                    torch.save(state, f)
            if is_best:
                state["best_clean_acc"] = state["val_clean_acc"]
                state["best_robust_acc"] = state["val_robust_acc"]
                with open(os.path.join(exp_path, f"model_best.pt"), "wb") as f:
                    torch.save(state, f)

            # log this epoch
            logger.info(utils.get_summary(train_meters + val_meters))
            logger.info(ENV)
    # No need to test last
    if not args.eval:
        checkpoint = utils.load_model(
            os.path.join(exp_path, "model_best"), backbone, optimizer
        )
        logger.info("Test best epoch {}".format(checkpoint["epoch"]))
    test_meters = test(
        False, test_model, test_text_features, test_adversary, testloader, device, args
    )
    logger.info(utils.get_summary(test_meters))
    meters = filter(lambda _: isinstance(_, utils.SamplesMeter), test_meters)
    logger.info("  ".join(m.summary() for m in meters))


if __name__ == "__main__":
    main()
