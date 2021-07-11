# Roberta Bayesian Hyperparameter Optimization with Wandb CV
 This is my experiment with multy-layer hyperparam. opt. using K-Folds.
 
<h3>Task&Data</h3>
 
- The task and the data are from the qualification round of AIIJC. In a nutshell, you need to determine whether there is any causal relationship between the two sentences in 5 different languages (multy-language binary classification). You can learn more about it [here](https://aiijc.com/en/task/1055/).
- The target metric was [MCC](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
 
<h3>Code</h3>

- The code is based on the [official Wandb CV example](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation) and [Simpletransformers Hyperparameter Optimization tutorial](https://towardsdatascience.com/hyperparameter-optimization-for-optimum-transformer-models-b95a32b70949).  
- To make the model, optimize and learn it, I used [Simpletransformers library](https://simpletransformers.ai/), which provides wide and usable set of settings for the most popular NLP models.
- I also used Wandb sweeps to track the optimization process.

<h3>Hyperparameters</h3>

You can find the list of hyp. in the `config.yaml` file. Those four starting with `layer_` denote network layer groups (Roberta-large has 24 encoder layers, so we have 4 groups with 6 layers each) to **set different learning rates** to prevent overfitting. Here is [simpletransformers reference](https://simpletransformers.ai/docs/tips-and-tricks/#custom-layer-parameters).

<h3>Results</h3>

The solution based only on the hyperparameter optimization (but for now unfortunately without K-Folds) is **top 7 of 131** at the leaderboard and the absolute metric is 0.485 (the best one is 0.549). Most importantly, this result was achieved with the help of **the basic model**, without ensemble techniques and other **exclusively competitive ways** to improve quality.

You now can check [this wandb report](https://wandb.ai/antivistrock/CVHPopt_Simpletransformers/reports/Roberta-Hyperparameter-Optimization--Vmlldzo4NDMxMzM?accessToken=18akkulp9jz8617my1cpxwvcufp6kwwc9nj9smb3i5ikw0elixc5v4hgehva8yma) and look at the meaningful dashboards. It **includes K-Fold** and MCC is mean of all 5 folds. So bayesian optimization updates by **the mean value of all data**.

<h3>How to reproduce</h3>

The easiest way is to run [this colab notebook](https://colab.research.google.com/drive/1Ogpztb--J8_1qAcyeHRJkGZq7LtSjoRF?usp=sharing)
