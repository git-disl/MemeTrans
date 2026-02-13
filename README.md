# MemeTrans

This repository contains the dataset and code for the paper: **"MemeTrans: A Dataset for Detecting High-Risk Memecoin Launches on
Solana"**

## Environment

- Python 3.9
- Conda recommended
- Install packages using:
``` sh
pip install -r requirements.txt
```

## Part 1: High-risk Memecoin Prediction

### Step 1: Train ML Models on the Generated Features & Labels

``` sh
cd MemeTrans/risk_prediction 
python ml_model_train.py --model rf
```

### Step 2: Evaluate Results in the Memecoin Selection Application

``` sh
python memecoin_selection.py
```


## Part 2: Generating the Dataset from Raw Data

Due to the large volume of the dataset (>100GB), we are currently preparing the
raw data and will release it soon via an external link.


## Q&A

If you have any questions, please open an issue or contact the corresponding author at: husihao26@gmail.com

We will respond as soon as possible.
# MemeTrans
# MemeTrans
# MemeTrans
