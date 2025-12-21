# hstu baseline pipeline
0. for autodl env, we need to remove conda sources
```bash
vim /root/.condarc
# remove anything except `- default`
```

1. set hstu env
```bash
cd baseline/hstu
chmod +x set_environment.sh
sh set_environment.sh
conda init
source ~/.bashrc
conda activate hstu_baseline
```

2. install packages
```bash
sh install_packages.sh
```

3. download data and preprocess
```bash
mkdir -p tmp/ && python preprocess_public_data.py
```

4. train model
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345
```

# experment setup
1. copy data
```bash
cd my_model
cp -r ../baseline/hstu/tmp/ .
```






# other setup
1. bash: less/tree
```bash
apt-get update
apt-get install less
apt-get install tree
```