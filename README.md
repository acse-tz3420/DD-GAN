# TSP-AI: Time Series Prediction with AI Methods for Fluid Flow

[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

This is the repository of ACSE9 individual project. TSP-AI is a python machine learning package to implement time series prediction of fluid flow.

## Table of Contents

- [About TSP-AI](#about-tsp-ai)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)


## About TSP-AI

This individual project applys time derivative method and create time series multi-step prediction model, using MLP, GAN or AAE, respectively. NIROM is constructed by applying POD to snapshots of numerical simulation for the flow past a cylinder. The time derivative method is applied to the prediction, and the difference between the result of directly predicting the next time level is compared with the time derivative method to assess its validity. These package include machine learning modules to data processing, model training and multi-step prediction with `Tensorflow` and `Keras`.

The multi-step predictive GAN draws on ideas from recent research [Predictive GAN](https://arxiv.org/abs/2105.07729). The predictive AAE based on [Predictive AAE](https://github.com/acse-zrw20/DD-GAN-AE). Work flows of these two models are shown below:

![PredGAN](https://github.com/acse-tz3420/TSP-AI/blob/main/images/PredGAN.png)


![PredAAE](https://github.com/acse-tz3420/TSP-AI/blob/main/images/PredAAE.png)


## Installation

To install TSP-AI, run the following commands:

```
# get the code
git clone https://github.com/acse-tz3420/TSP-AI.git
cd TSP-AI

# install pre-requisites
 $ conda env create -f environment.yml 
 $ conda activate tspai
 $ python -m ipykernel install --user --name=python3
```


### Data Downloading

To download the data, first go to [kaggle](https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials which you must upload to colab. Then you must move this file to your root directory and request a forced update of kaggle in pip to download your data correctly. Simply run:

```
bash download_data.sh
```


## Usage

Use `python main.py` in `acse_softmax` to train model and use `--help` option to see all the parameters that can be changed.

```bash
usage: Covid_CT_Classifier_Softmax [-h] [-version] [-s S] [-lr LR] [-m M] [-bs BS] [-ts TS] [-e E]

optional arguments:
  -h, --help  show this help message and exit
  -version    show program's version number and exit
  -s S        seed
  -lr LR      learning rate
  -m M        momentum
  -bs BS      batch size
  -ts TS      test batch size
  -e E        epoch

```

Or you can choose to use `X-RAY-Classifier.ipynb`, which includes all functional parts, as well as user guidence, process of tuning parameters and strategy to select the final submission.


## Documentation

To get documentation of TSP-AI, open `index.html` in `docs/html` after installation.


## Contributing

Feel free to dive in! [Open an issue](https://github.com/acse-tz3420/TSP-AI/issues/new) or submit PRs.

### Contributors

The author of this project is Tianyi Zhao. This project exists thanks to my supervisors, Dr. Claire Heaney and Prof. Christopher Pain, and all the people in DD-GAN group.


## License

[MIT](LICENSE) © acse-2020 Tianyi Zhao
