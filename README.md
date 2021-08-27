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

This individual project applys time derivative method and create time series multi-step prediction model, using MLP, GAN or AAE, respectively. NIROM is constructed by applying POD to snapshots of numerical simulation for the flow past a cylinder. The time derivative method is applied to the prediction, and the difference between the result of directly predicting the next time level is compared with the time derivative method to assess its validity. These package include machine learning modules to data processing, model training and multi-step prediction with `Tensorflow` and `Keras`. The model uses GPU for acceleration during training, but this is not a necessary item.

The multi-step predictive GAN draws on ideas from recent research [Predictive GAN](https://arxiv.org/abs/2105.07729). The predictive AAE is based on [Predictive AAE](https://github.com/acse-zrw20/DD-GAN-AE). Work flows of these two models are shown below:

![PredGAN](https://github.com/acse-tz3420/TSP-AI/blob/main/images/PredGAN.png)


![PredAAE](https://github.com/acse-tz3420/TSP-AI/blob/main/images/PredAAE.png)


## Installation

To install TSP-AI, run the following commands:

```
# get the code
git clone https://github.com/acse-tz3420/TSP-AI.git
cd ./TSP-AI

# install pre-requisites
 $ conda env create -f environment.yml 
 $ conda activate tspai
 $ python -m ipykernel install --user --name=python3
```


## Usage

Usage examples can be found in `examples` folder. Simply run the notebooks to train the model, make prediction and see results. The notebooks in this folder contain the final results included in [Final Report](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-tz3420/blob/main/TianyiZhao_ACSE9_FinalReport.pdf).

### Data Pre-processing and Reconstrction

The flow past a cylinder data used as test case is in `data` folder. For separated pre-processing and post-processing modules, the source code emanates from Dr. Heaney and was compiled by Zef Wolffs and Jon Atli Tomasson in DD-GAN team, with some modifications from Tianyi Zhao. Please see `preprocessing` folder for futher instructions.


### Working Logs

Folder `worklog` contains the working logs, weekly meeting notes, and other results mentioned in [Final Report](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-tz3420/blob/main/TianyiZhao_ACSE9_FinalReport.pdf). These could be a proof of the author's continuous work during the three months.


## Documentation

Find the [documentation](https://github.com/acse-tz3420/TSP-AI/blob/main/docs/build/html/index.html) of TSP-AI. Open the `index.html` file after installation.


## Contributing

Feel free to dive in! [Open an issue](https://github.com/acse-tz3420/TSP-AI/issues/new) or submit PRs.

### Contributors

The author of this project is Tianyi Zhao. This project exists thanks to my supervisors, Dr. Claire Heaney and Prof. Christopher Pain, and all the people in DD-GAN group.


## License

[MIT](LICENSE) © acse-2020 Tianyi Zhao
