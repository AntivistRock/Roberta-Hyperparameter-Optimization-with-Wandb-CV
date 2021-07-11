import os
import multiprocessing
import collections
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.model_selection import KFold
import pandas as pd
import json
import tqdm
import wandb
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_train(reversed=False):
  file = open('/content/39')
  train_data = []
  for str_json in list(file):
    item = json.loads(str_json)
    prem = item['premise']
    hyp = item['hypothesis']
    label = item['label']
    if reversed:
      train_data.append([hyp, prem, label])
    else:
      train_data.append([prem, hyp, label])
  return train_data

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num",
                       "sweep_id",
                       "sweep_run_name",
                       "config",
                       "train_df",
                       "val_df"
                       )
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("mcc"))

model_args = ClassificationArgs(
        dataloader_num_workers=4,
        evaluate_during_training=True,
        eval_batch_size=6,
        train_batch_size=4,
        overwrite_output_dir=True,
        max_seq_length=306,
        no_save=True,
        no_cache=True,
        labels_list=["not_entailment", "entailment"],
        silent=True
    )

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(sweep_q, worker_q):
    reset_wandb_env()
    worker_data = worker_q.get()
    train_df = worker_data.train_df
    val_df = worker_data.val_df
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.num)
    config = worker_data.config
    run = wandb.init(
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config,
    )
    args = {
        key: value
        for key, value in wandb.config.as_dict().items()
        if key != "_wandb"
    }

    cleaned_args = {}
    layer_params = []
    param_groups = []
    for key, value in args.items():
        if key.startswith("layer_"):
            layer_keys = key.split("_")[-1]

            start_layer = int(layer_keys.split("-")[0])
            end_layer = int(layer_keys.split("-")[-1])

            for layer_key in range(start_layer, end_layer):
                layer_params.append(
                    {"layer": layer_key, "lr": value,}
                )
        elif key.startswith("params_"):
            params_key = key.split("_")[-1]
            param_groups.append(
                {
                    "params": [params_key],
                    "lr": value,
                    "weight_decay": model_args.weight_decay
                    if "bias" not in params_key
                    else 0.0,
                }
            )
        else:
            cleaned_args[key] = value
    cleaned_args["custom_layer_parameters"] = layer_params
    cleaned_args["custom_parameter_groups"] = param_groups
    model_args.update_from_dict(cleaned_args)

    model = ClassificationModel(
        "xlmroberta",
        "vicgalle/xlm-roberta-large-xnli-anli",
        use_cuda=True,
        args=model_args,
        sweep_config=run.config,
    )
    print("Current training is started.")
    # Train the model
    model.train_model(
        train_df,
        eval_df=val_df,
    )
    print(f"Model-{run_name} training is ended.")
    wandb.join()
    #sweep_q.put(WorkerDoneData(result['mcc']))


def main():
    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=1140, shuffle=True)
    train_data = get_train()
    train_data = pd.DataFrame(train_data)
    train_data.columns = ["text_a", "text_b", "labels"]
    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    for num in range(num_folds):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    #sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    metrics = []
    for num, (train_index, val_index) in enumerate(kf.split(train_data)):
        worker = workers[num]
        train_df = train_data.iloc[train_index]
        val_df = train_data.iloc[val_index]
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
                train_df=train_df,
                val_df=val_df
            )
        )
        result = None
        # wait for worker to finish
        worker.process.join()
        with open('/content/outputs/eval_results.txt') as F:
            file = F.readlines()
            result = float(file[1].split()[-1])
        print('Curr mcc is:', result)
        metrics.append(result)
    sweep_run.log(dict(mcc=sum(metrics) / len(metrics)))
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)

if __name__ == "__main__":
    main()
