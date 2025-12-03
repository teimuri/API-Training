import os
from dotenv import load_dotenv
import time
from pathlib import Path

import torch
from clearml import Task
from transformers import TrainerCallback

from ml_trainer import AutoTrainer
from aipmodel.model_registry import MLOpsManager
from data.sdk.download_sdk import s3_download

torch._dynamo.config.disable = True

# import the torch callback for checkpointing
# import os
# import shutil

#‍‍‍‍ let's plan this out
# Three parts: one data another is the model and the new one is the retrying & resuming
# for the model it's simple you load and save it then give the path? or if it's retrying you do the resume_checkpoint
# then another thing is 
# --------- ClearML task initialization --------
task = Task.init(
    project_name="API training",  # Name of the ClearML project
    task_name=f"API Training",  # Name of the task
    # task_type=Task.TaskTypes.optimizer,  # Type of the task (could also be "training", "testing", etc.)
    reuse_last_task_id=False  # Whether to reuse the last task ID (set to False for a new task each time)
)
## ====================== Data Registry =========================
load_dotenv()

user_management_api = os.getenv("USER_MANAGEMENT_API")
clearml_api_host = os.getenv("CLEARML_API_HOST")
s3_endpoint_url = os.getenv("CEPH_ENDPOINT_URL")

print(user_management_api, clearml_api_host, s3_endpoint_url)

data_model_reg_cfg= {
    'clearml_username': 'mlopsuser03',
    'token': 'default'
}

task.connect(data_model_reg_cfg, name='model_data_cfg')

print("Current ClearML Task ID:", task.id)



# --------- fetch model from model registry --------
manager = MLOpsManager(
    # user_name=data_model_reg_cfg['clearml_username'],
    user_token=data_model_reg_cfg['token'],
    # user_management_url=user_management_api,
    # clearml_api_host=clearml_api_host,
    # s3_endpoint_url=s3_endpoint_url,
)

class PrintSaveDirCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        import os
        print(f"\n[Callback] Model saved to: {args.output_dir}")
        print("[Callback] Files inside:")

        for f in os.listdir(args.output_dir):
            print("  -", f)

        model_name = f"checkpoint-{task.id}"
        try:
            model_id = manager.get_model_id_by_name(model_name)
            if model_id:
                print(f"[Callback] Model with name '{model_name}' already exists in registry with ID: {model_id}")
                manager.delete_model(model_id=model_id)
                print(f"[Callback] Deleted existing model with ID: {model_id}")
            
        except Exception as e:
            print(f"[Callback] Error fetching model ID for {model_name}: {e}")
            print("[Callback] Proceeding to add the model as new.")


        try:
            model_id = manager.add_model(
                source_type="local",
                source_path=args.output_dir,
                model_name=model_name,
            )
            print(f"[Callback] Model uploaded to registry with ID: {model_id}\n")
        except Exception as e:
            print(f"[Callback] Failed to upload model '{model_name}': {e}")
        


config = {
    "task": "llm_finetuning",
    # "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct_1762009294_1762010845",

    # -----------------------------
    # DATASET CONFIG
    # -----------------------------
    "system_prompt": "You are a helpful assistant.",
    "dataset_config": {
        "source": "medical_qaa",
        "format_fn": None,
        # "format_fn": "default",
        "test_size": None,
    },

    # -----------------------------
    # TRAINER CONFIG
    # -----------------------------
    "trainer_config": {
        "dataset_text_field": "text",
        "batch_size": 2, # *
        # "epochs": 1, # *
        "epochs": 1, # *
        "learning_rate": 1e-4, # *
        
        "save_steps": 0.5,
        "save_strategy": "epoch",

        

        "optim": "adamw_8bit",
        "save_total_limit": 1,
        "output_dir": "./model",
        "resume_from_checkpoint": None,
        "callbacks": [PrintSaveDirCallback()],

        "load_model": None,  # set to True to load model from model registry
        "save_model": None,  # set to True to save model to model registry
    },
}
task.connect(config)

print(config)

model_reg = config["model_name"]
if config["trainer_config"]["load_model"] is "False" or config["trainer_config"]["load_model"] is "false" or config["trainer_config"]["load_model"] == "":
    config["trainer_config"]["load_model"] = None
    
# --------------     to load model -----------------
if config["trainer_config"]["load_model"] is not None: 
    model_id = manager.get_model_id_by_name(model_reg)
    manager.get_model(
        model_name= model_reg,  # or any valid model ID
        local_dest="."
    )
    model_dir = f'./{model_id}/'

    # Find the first folder inside model_dir
    subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]

    if not subfolders:
        print(f"No checkpoint folders found in {model_dir}")

    # You can choose the first one or specify logic (e.g., latest modified)

    if subfolders:
        checkpoint_folder = subfolders[0]  # or sorted(subfolders)[-1] for the last alphabetically
        config["model_name"] = f'./{model_id}/{checkpoint_folder}/'
        print(f"Checkpoint folder found: {checkpoint_folder}")
        print(f"Model path set to: {config['model_name']}")
    else:
        config["model_name"] = f'./{model_id}/'
        print(f"Model path set to: {config['model_name']}")


if config['trainer_config']["resume_from_checkpoint"] is not None:

    task_id = config['trainer_config']["resume_from_checkpoint"]

    checkpoint_name = f"checkpoint-{task_id}"
    print(f"Resuming from task ID: {task_id}")

    model_id = manager.get_model_id_by_name(checkpoint_name)
    manager.get_model(
        model_name=checkpoint_name,  # or any valid model ID
        local_dest="."
    )

    checkpoint_dirs = [f for f in os.listdir(model_id) if os.path.isdir(os.path.join(model_id, f))]

    if not checkpoint_dirs:
        print(f"No checkpoint folders found in {model_id}")
        config["trainer_config"]["resume_from_checkpoint"] = f'./{model_id}/'
        print(f"Resume checkpoint path set to: {config['trainer_config']['resume_from_checkpoint']}")
    else:
        # Option 1: take the first found folder
        # checkpoint_folder = checkpoint_dirs[0]

        # Option 2: or take the latest alphabetically (often newest)
        checkpoint_folder = sorted(checkpoint_dirs)[-1]

        config["trainer_config"]["resume_from_checkpoint"] = f'./{model_id}/{checkpoint_folder}/'
        print(f"Checkpoint folder found: {checkpoint_folder}")
        print(f"Resume checkpoint path set to: {config['trainer_config']['resume_from_checkpoint']}")
        

s3_download(
        dataset_name=config["dataset_config"]["source"],
        absolute_path=Path(__file__).parent/"dataset",
        token=data_model_reg_cfg['token'],
        user_management_url=user_management_api,
        clearml_api_host=clearml_api_host,
        s3_endpoint_url=s3_endpoint_url,
        # user_name=data_model_reg_cfg['clearml_username'],
    )

config["dataset_config"]["source"] = s3_download

# absolute_path = Path(__file__).parent / "dataset" / config["dataset_config"]["source"]

# files = list(absolute_path.rglob("*.[jc][so][nv]*"))  # matches .json or .csv

# # Or, more explicitly:
# files = [f for f in absolute_path.rglob("*") if f.suffix in [".json", ".csv"]]

# # Print absolute paths
# for file_path in files:
#     print(file_path.resolve())
#     file_path = file_path.resolve()

# # Connect hyperparameters and other configurations to the ClearML task



# config["dataset_config"]["source"] = file_path


trainer = AutoTrainer(config=config)

trainer.run()

if config["trainer_config"]["save_model"] is not None:
    trainer.model.save("./full_model_save/")
    local_model_id = manager.add_model(
        source_type="local",
        source_path="full_model_save/",
        model_name = model_reg + "_" + str(int(time.time())),
    )
