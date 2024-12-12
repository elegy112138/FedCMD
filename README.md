# Towards Optimal Customized Architecture for Heterogeneous Federated Learning with Contrastive Cloud-Edge Model Decoupling

## 1. Abstract
Federated learning, as a promising distributed learning paradigm, enables collaborative training of a global model across multiple network edge clients without the need for central data collecting. However, the heterogeneity of edge data distribution will draft the model towards the local minima, which can be distant from the global optimum. This phenomenon called “client-draft” often leads to slow convergence and substantial communication overhead. To tackle these issues, we propose a novel federated learning framework called **FedCMD** with a contrastive Cloud-edge model decoupling to separate deep neural networks into a body for capturing shared representations in the Cloud and a personalized head for migrating data heterogeneity in the edges. Departing from the traditional approach of rigidly assigning the last layer as the personalized head and pre-output layers as the body, we explore the performance gains of selecting different neural network layers as the personalized head and find that utilizing the last layer is not always optimal. This observation inspired us to investigate a representation method to capture the heterogeneity of data distributions of each client and leverage the knowledgeization toward optimal personalized layer selection. Therefore, we utilize the low-dimensional representation of each layer to contrast feature distribution transfer and introduce a Wasserstein-based layer selection method, aimed at identifying the best-match layer for personalization. Additionally, a weighted global aggregation algorithm is proposed based on the selected personalized layer for the practical application of **FedCMD**. Extensive experiments on ten benchmarks demonstrate the efficiency and superior performance of our solution compared with nine state-of-the-art solution.

This work is detailed further in our paper published in IEEE Transactions on Computers:
> Chen, X., Du, T., Wang, M., Gu, T., Zhao, Y., Kou, G., Xu, C., & Wu, D. O. (2024). Towards Optimal Customized Architecture for Heterogeneous Federated Learning with Contrastive Cloud-Edge Model Decoupling. *IEEE Transactions on Computers*, 1-14. https://doi.org/10.1109/TC.2024.3514302

## 2. Framework
<div>
     <img src="/images/FedCMD.png" alt="Framework">
</div>

The above figure presents the diagram of FedCMD. We consider the Cloud-edge cooperation system, in which two types of nodes are involved, including one Cloud server, and multiple edge clients. FedCMD contains two major phases including **the personalized layer selection phase** and the heterogeneous federated learning phase with the Cloud-edge model decoupling. In personalized layer selection phase, the Cloud-edge system employs the standard federated learning such as FedAvg and utilizes the contrastive layer selection mechanism to collaboratively elect the personalized layer. Thus, the selection proceeds as follows:
- **Global parameters broadcasting:** The central server initializes the global model parameter and distributes them to all edge clients.
- **Local model updating:** After receiving the global parameter, each client updates their local model and trains the model using the local data.
- **Local Layer Selection:** Based on the criteria of the personalized layer selection, each edge client selects a personalized layer from their model during the training.
- **Communication of local gradients:** Edge clients send updates of their local model parameters and their chosen personalized layer back to the Cloud server.
- **Cloud-side processing:** The Cloud server applies the standard federated aggregation to update the parameter of the global model and determine the layer selected by the most number of edge clients as the personalized layer.

Once this personalized layer is determined, it remains fixed throughout **the heterogeneous federated learning phase**. The personalized layer’s parameters do not participate in global aggregation and are updated locally on the edge client side. In contrast, the other layers of the model parameters are updated through a weighted federated aggregation, the specifics of which will be detailed in the algorithm design section. The process of the federated learning phase can be outlined as:
- **Global parameters broadcasting:** The central Cloud server transmits the global model parameter, excluding the personalized layer, to edge clients.
- **Local model updating:** Upon receiving the global parameter, each edge client updates their local model and trains the model using the local data.
- **Communication of local gradients:** Edge clients send updates of their local model parameters back to the Cloud server.
- **Cloud-side processing:** The Cloud server utilizes the personalized layer parameters to calculate the weight matrix and implements weighted federated aggregation to update the parameters of the body layers.

## 3. Experimental Results

Our extensive experiments demonstrate the efficacy of FedCMD across a variety of settings and compare it against other state-of-the-art federated learning methods.

### 3.1 Performance Overview

The following tables summarize the average model accuracy across different datasets and configurations:

#### Table I: Average Model Accuracy for α = 0.1
![Average Model Accuracy for α = 0.1](/images/TableVI.png)

#### Table II: Average Model Accuracy for α = 0.5
![Average Model Accuracy for α = 0.5](/images/TableVII.png)

#### Table III: Average Model Accuracy for α = 1.0
![Average Model Accuracy for α = 1.0](/images/TableVIII.png)

### 3.2 Accuracy Trends Over Epochs

provides a comparison of accuracy trends over epochs for FedCMD against other methods:

![Accuracy Comparison of FedCMD and other methods](/images/Figure5.png)

### 3.3 Impact of Join Ratio and Number of Rounds

Figure and Table IX illustrate how different join ratios and the number of layer selection rounds influence the accuracy across ten datasets:

![Accuracy comparison for different join ratios](/images/Figure7.png)

#### Table IV: Accuracy with Different Rounds of Layer Selection Phase
![Accuracy with Different Rounds of Layer Selection Phase](/images/TableIX.png)


## 3. Installation

### Environment Preparation 🚀

#### PyPI 🐍
📢 Note that FedCMD needs `3.10 <= python < 3.12`. I suggest you to checkout your python version before installing packages by pip.
```
pip install -r requirements.txt
```

#### Conda 💻
```
conda env create -f environment.yml
```

#### Poetry 🎶

**At China mainland**
```
poetry install
```

**Not at China mainland**
```
sed -i "26,30d" pyproject.toml && poetry lock --no-update && poetry install
```

#### Docker 🐳

**At China mainland**
```
docker build -t FedCMD .
```

**Not at China mainland**
```
docker build \
-t FedCMD \
--build-arg IMAGE_SOURCE=karhou/ubuntu:basic \
--build-arg CHINA_MAINLAND=false \
.
```

### Download FedCMD
You can download FedCMD from Github:
```
git clone https://github.com/elegy112138/FedCMD
```

## 4. How to use

### Folders
<ul>
  <li><strong>./data/:</strong> This folder is used to store the datasets, where the file [`./data/generate_data.py`](https://github.com/elegy112138/FedCMD/blob/main/data/generate_data.py) is used to divide and generate the datasets.</li>
  <li><strong>./src/:</strong> This folder contains configuration, client-side and server-side files, including fedcmd and other baseline methods, for easy user changes.</li>
  <li><strong>./Experiments/:</strong> This folder holds all the results of our experiments.</li>
</ul>

### Datasets

#### Datasets Partition

```shell
# partition the CIFAR-10 according to Dir(0.1) for 100 clients
cd data
python generate_data.py -d cifar10 -a 0.1 -cn 100
cd ../
```

#### Supported Datasets

Regular Image Datasets

- *MNIST* (1 x 28 x 28, 10 classes)

- *CIFAR-10/100* (3 x 32 x 32, 10/100 classes)

- *EMNIST* (1 x 28 x 28, 62 classes)

- *FashionMNIST* (1 x 28 x 28, 10 classes)

- [*Syhthetic Dataset*](https://arxiv.org/abs/1812.06127)

- [*FEMNIST*](https://leaf.cmu.edu/) (1 x 28 x 28, 62 classes)

- [*CelebA*](https://leaf.cmu.edu/) (3 x 218 x 178, 2 classes)

- [*SVHN*](http://ufldl.stanford.edu/housenumbers/) (3 x 32 x 32, 10 classes)

- [*USPS*](https://ieeexplore.ieee.org/document/291440) (1 x 16 x 16, 10 classes)

- [*Tiny-ImageNet-200*](https://arxiv.org/pdf/1707.08819.pdf) (3 x 64 x 64, 200 classes)

- [*CINIC-10*](https://datashare.ed.ac.uk/handle/10283/3192) (3 x 32 x 32, 10 classes)

- [*DomainNet*](http://ai.bu.edu/DomainNet/) (3 x ? x ?, 345 classes)
  - Go check [`data/README.md`](https://github.com/elegy112138/FedCMD/blob/main/data/README.md) for the full process guideline 🧾.

Medical Image Datasets

- [*COVID-19*](https://www.researchgate.net/publication/344295900_Curated_Dataset_for_COVID-19_Posterior-Anterior_Chest_Radiography_Images_X-Rays) (3 x 244 x 224, 4 classes)

- [*Organ-S/A/CMNIST*](https://medmnist.com/) (1 x 28 x 28, 11 classes)

### Easy Run 🏃‍♂️

FedCMD and other baseline methods are inherited from `FedAvgServer` and `FedAvgClient`. If you wanna figure out the entire workflow and detail of variable settings, go check [`./src/server/fedavg.py`](https://github.com/elegy112138/FedCMD/blob/main/src/server/fedavg.py) and [`./src/client/fedavg.py`](https://github.com/elegy112138/FedCMD/blob/main/src/client/fedavg.py).

```shell
# run FedAvg on CIFAR-10 with default settings.
cd src/server
python fedcmd.py -d cifar10
```

About methods of generating federated dastaset, go check [`data/README.md`](https://github.com/elegy112138/FedCMD/blob/main/data/README.md) for full details.


#### Monitor 📈 (recommended 👍)
1. Run `python -m visdom.server` on terminal.
2. Run `src/server/fedcmd.py --visible 1`
3. Go check `localhost:8097` on your browser.
   
### Generic Arguments 🔧

📢 All generic arguments have their default value. Go check `get_fedavg_argparser()` in [`FedCMD/src/server/fedavg.py`](https://github.com/elegy112138/FedCMD/blob/main/src/server/fedavg.py) for full details of generic arguments.

About the default values and hyperparameters of advanced FL methods, go check corresponding `FedCMD/src/server/fedcmd.py` for full details.
| Argument                       | Description                                                                                                                                                                                                                                                                                                                               |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset`                    | The name of dataset that experiment run on.                                                                                                                                                                                                                                                                                               |
| `--model`                      | The model backbone experiment used.                                                                                                                                                                                                                                                                                                       |
| `--seed`                       | Random seed for running experiment.                                                                                                                                                                                                                                                                                                       |
| `--join_ratio`                 | Ratio for (client each round) / (client num in total).                                                                                                                                                                                                                                                                                    |
| `--global_epoch`               | Global epoch, also called communication round.                                                                                                                                                                                                                                                                                            |
| `--local_epoch`                | Local epoch for client local training.                                                                                                                                                                                                                                                                                                    |
| `--finetune_epoch`             | Epoch for clients fine-tunning their models before test.                                                                                                                                                                                                                                                                                  |
| `--test_gap`                   | Interval round of performing test on clients.                                                                                                                                                                                                                                                                                             |
| `--eval_test`                  | Non-zero value for performing evaluation on joined clients' testset before and after local training.                                                                                                                                                                                                                                      |
| `--eval_train`                 | Non-zero value for performing evaluation on joined clients' trainset before and after local training.                                                                                                                                                                                                                                     |
| `--local_lr`                   | Learning rate for client local training.                                                                                                                                                                                                                                                                                                  |
| `--momentum`                   | Momentum for client local opitimizer.                                                                                                                                                                                                                                                                                                     |
| `--weight_decay`               | Weight decay for client local optimizer.                                                                                                                                                                                                                                                                                                  |
| `--verbose_gap`                | Interval round of displaying clients training performance on terminal.                                                                                                                                                                                                                                                                    |
| `--batch_size`                 | Data batch size for client local training.                                                                                                                                                                                                                                                                                                |
| `--use_cuda`                   | Non-zero value indicates that tensors are in gpu.                                                                                                                                                                                                                                                                                         |
| `--visible`                    | Non-zero value for using Visdom to monitor algorithm performance on `localhost:8097`.                                                                                                                                                                                                                                                     |
| `--global_testset`             | Non-zero value for evaluating client models over the global testset before and after local training, instead of evaluating over clients own testset. The global testset is the union set of all client's testset. *NOTE: Activating this setting will considerably slow down the entire training process, especially the dataset is big.* |
| `--save_log`                   | Non-zero value for saving algorithm running log in `FedCMD/out/${algo}`.                                                                                                                                                                                                                                                                |
| `--straggler_ratio`            | The ratio of stragglers (set in `[0, 1]`). Stragglers would not perform full-epoch local training as normal clients. Their local epoch would be randomly selected from range `[--straggler_min_local_epoch, --local_epoch)`.                                                                                                              |
| `--straggler_min_local_epoch`  | The minimum value of local epoch for stragglers.                                                                                                                                                                                                                                                                                          |
| `--external_model_params_file` | (New feature ✨) The relative file path of external (pretrained) model parameters (`*.pt`). e.g., `../../out/FedAvg/mnist_100_lenet5.pt`. Please confirm whether the shape of parameters compatible with the model by yourself. ⚠ This feature is enabled only when `unique_model=False`, which is pre-defined by each FL method.          |
| `--save_model`                 | Non-zero value for saving output model(s) parameters in `FedCMD/out/${algo}`.  The default file name pattern is `${dataset}_${global_epoch}_${model}.pt`.                                                                                                                                                                               |
| `--save_fig`                   | Non-zero value for saving the accuracy curves showed on Visdom into a `.jpeg` file at `FedCMD/out/${algo}`.                                                                                                                                                                                                                             |
| `--save_metrics`               | Non-zero value for saving metrics stats into a `.csv` file at `FedCMD/out/${algo}`.                                                                                                                                                                                                                                                     |

## Acknowledge
This work is based on previous <strong>pFedSim</strong> studies, which are the [`source`](https://github.com/KarhouTam/FL-bench) of the project documents.
