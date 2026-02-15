import os
import re
import matplotlib.pyplot as plt

pattern = r"Epoch (?P<epoch>\d+): TrainLoss=(?P<train_loss>[\d.]+), EvalLoss=(?P<eval_loss>[\d.]+), Acc=(?P<acc>[\d.]+), BinaryAcc=(?P<bin_acc>[\d.]+), AUC=(?P<auc>[\d.]+)"

log_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_log')
output_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

baseline_dir = os.path.join(log_home, 'baseline')
ablation_dir = os.path.join(log_home, 'ablation')
exp_dir = os.path.join(log_home, 'exp')

epoch = {}
loss = {}
acc = {}
auc = {}


def extract_data(dir, prefix):
    for fname in os.listdir(dir):
        exp = fname.replace('.log', '').replace(prefix, '')
        exp_name = '_'.join(exp.split('_')[:-1])
        dataset_name = exp.split('_')[-1]
        _epoch = []
        _loss = []
        _acc = []
        _auc = []
        with open(os.path.join(dir, fname), 'r') as f:
            for line in f:
                if '[Eval] End of Epoch ' in line:
                    try:
                        match = re.search(pattern, line)
                        results = {k: float(v) for k, v in match.groupdict().items()}
                        _epoch.append(results['epoch'])
                        _loss.append(results['eval_loss'])
                        _acc.append(results['acc'])
                        _auc.append(results['auc'])
                    except:
                        print(f'err while stracting {exp_name}, {dataset_name}, epoch {_epoch[-1]+1}')
        if dataset_name not in loss.keys():
            loss[dataset_name] = {}
            acc[dataset_name] = {}
            auc[dataset_name] = {}
            epoch[dataset_name] = {}
        loss[dataset_name][exp_name] = _loss
        acc[dataset_name][exp_name] = _acc
        auc[dataset_name][exp_name] = _auc
        epoch[dataset_name][exp_name] = _epoch
    return



extract_data(baseline_dir, 'baseline_')
extract_data(ablation_dir, 'ablation_')
extract_data(exp_dir, 'exp_')



def draw_exp(x, epoch, y_label):
    for dataset_name in x.keys():
        figure, axs = plt.subplots(figsize = (10, 5))
        for exp_name in x[dataset_name].keys():
            axs.plot(epoch[dataset_name][exp_name], x[dataset_name][exp_name], label=exp_name)

        axs.set_xlabel('epoch')
        axs.set_ylabel(f'{y_label}')
        axs.set_title(f'{dataset_name}')
        plt.legend(title = 'label')
        plt.savefig(f'{output_home}/exp_{dataset_name}_{y_label}.png')


draw_exp(loss, epoch, 'loss')
draw_exp(acc, epoch, 'acc')
draw_exp(auc, epoch, 'auc')