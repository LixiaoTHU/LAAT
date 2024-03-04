import os
import torch


def eval_aa(model, test_loader, epoch, epsilon, fname, args):
    if epoch is None:
        aa_save_dir = os.path.join(fname, "model_best_adv_inputs")
    else:
        aa_save_dir = os.path.join(fname, f"model_{epoch}_adv_inputs")
    if not os.path.exists(aa_save_dir):
        os.makedirs(aa_save_dir)
    aa_log_path = os.path.join(aa_save_dir, "log.txt")
    from autoattack import AutoAttack

    adversary = AutoAttack(model, norm="Linf", eps=epsilon, log_path=aa_log_path)
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    adv_complete = adversary.run_standard_evaluation(
        x_test, y_test, bs=args.batch_size_test
    )
    torch.save(
        {"adv_complete": adv_complete},
        "{}/{}_{}_1_{}_eps_{:.5f}.pth".format(
            aa_save_dir, "aa", "standard", adv_complete.shape[0], epsilon
        ),
    )
