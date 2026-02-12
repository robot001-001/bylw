import pandas as pd
import matplotlib.pyplot as plt

data_baseline = f'record/speed_exp_baseline.log'
data_bsa = f'record/speed_exp_bsa.log'
data_bsa_64 = f'record/speed_exp_bsa_64.log'

def read_data(path):
    ret = {
        'seq_len': [],
        'Time': [],
        'mem': []
    }
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'seq_len' in line:
                ret['seq_len'].append(float(line.split(': ')[1]))
            elif 'Time' in line:
                ret['Time'].append(float(line.split(': ')[1].split(' ')[0]))
            elif 'Kernel Overhead' in line:
                ret['mem'].append(float(line.split(': ')[1].strip().split(' ')[0]))
    assert len(ret['seq_len'])==len(ret['Time']), f"{len(ret['seq_len'])} {len(ret['Time'])} {len(ret['mem'])}"
    assert len(ret['seq_len'])==len(ret['mem']), f"{len(ret['seq_len'])} {len(ret['Time'])} {len(ret['mem'])}"
    return ret

data_baseline = read_data(data_baseline)
data_bsa = read_data(data_bsa)
data_bsa_64 = read_data(data_bsa_64)

figure, axs = plt.subplots(figsize = (10, 5), nrows=1, ncols=2)

axs[0].plot(data_baseline['seq_len'], data_baseline['Time'], label='baseline')
# axs[0].plot(data_bsa['seq_len'], data_bsa['Time'], label='bsa_32')
axs[0].plot(data_bsa_64['seq_len'], data_bsa_64['Time'], label='bsa')

axs[0].set_xlabel('seq lens')
axs[0].set_ylabel('time cost(ms)')
axs[0].set_title('time cost stat')


axs[1].plot(data_baseline['seq_len'], data_baseline['mem'], label='baseline')
# axs[1].plot(data_bsa['seq_len'], data_bsa['mem'], label='bsa_32')
axs[1].plot(data_bsa_64['seq_len'], data_bsa_64['mem'], label='bsa')

axs[1].set_xlabel('seq lens')
axs[1].set_ylabel('memory usage(MB)')
axs[1].set_title('memory cost stat')




plt.legend(title = 'label')
plt.savefig('output/speed_exp.png')