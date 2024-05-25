## System requirements

目前仅支持 ubuntu 系统，我们不保证能够在其他系统上完成配置。

如有必要，可能需要安装以下库：

``` shell
sudo apt install build-essential libtinfo5
```

## Environment setup

#### 1. Install conda or miniconda

如果您的计算机中已经安装了 conda 或 miniconda，那么则跳过这一步。

否则，我们使用 conda 来创建程序运行使用的虚拟环境，您可以访问 miniconda 官网进行 miniconda 的安装，或者直接执行以下脚本：

``` shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

#### 2. Clone this repo

略

``` shell
```

#### 3. Set up conda environment

依次执行以下代码即可，第一次执行 `pip install compiler_gym==0.2.5` 时会报错，但是忽略该错误即可，最终 `compiler_gym` 仍然能够成功安装。

``` shell
conda env create -f environment.yml
conda activate bs4cpo
pip install pip==22.2 setuptools==59.6.0
pip install compiler_gym==0.2.5
```