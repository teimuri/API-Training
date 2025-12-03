
import os
from dotenv import load_dotenv
import time
from pathlib import Path

from clearml import Task

from ml_trainer import AutoTrainer
from aipmodel.model_registry import MLOpsManager
from data.sdk.download_sdk import s3_download

# --------- ClearML task initialization --------
task = Task.init(
    project_name="Local API training",  # Name of the ClearML project
    task_name=f"API Training",  # Name of the task
    # task_type=Task.TaskTypes‍‍.optimizer,  # Type of the task (could also be "training", "testing", etc.)
    reuse_last_task_id=False  # Whether to reuse the last task ID (set to False for a new task each time)
)

load_dotenv()

user_management_api = os.getenv("USER_MANAGEMENT_API")
clearml_api_host = os.getenv("CLEARML_API_HOST")
s3_endpoint_url = os.getenv("CEPH_ENDPOINT_URL")

data_model_reg_cfg= {
    'clearml_username': 'mlopsuser03',
    'token': 'default'
}


task.connect(data_model_reg_cfg, name='model_data_cfg')

print(data_model_reg_cfg)

print("Current ClearML Task ID:", task.id)


# --------- fetch model from model registry --------
manager = MLOpsManager(
    # user_name=data_model_reg_cfg['clearml_username'],
    user_token=data_model_reg_cfg['token'],
)


class ModelRegistryCheckpoint():
    def on_save(self, path):
        path = 'checkpoint'
        import os
        print(f"\n[Callback] Model saved to: {path}")
        print("[Callback] Files inside:")

        # for f in os.listdir(path):
        #     print("  -", f)

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
                source_path=path,
                model_name=model_name,
            )
            print(f"[Callback] Model uploaded to registry with ID: {model_id}\n")
        except Exception as e:
            print(f"[Callback] Failed to upload model '{model_name}': {e}")

#----------------- main config ----------------
config = {
        "task": "image_classification",
        "model_name": "model registry",

        "dataset_config": {
            "source": "datasetname",  # !Required
            "batch_size": 32,                    # !Required
            "split_ratio": None,                 # !Required
            "transform_config": {
                "resize": (32, 32),
                # "horizontal_flip": True,
                # "normalization": {
                #     "mean": [0.4914, 0.4822, 0.4465],
                #     "std": [0.2023, 0.1994, 0.2010],
                # },
            },
        },
        "model_config": {
            "num_classes": 2,          # !Required
            "input_channels": 3,       # !Required
            "input_size": (32, 32),  
            "type": "timm",            
            "name": "resnet18",        # !Required
            "pretrained": True,        
            },
        
        "trainer_config": {
            "lr": 1e-2,                # *
            "load_model": None,        # *
            "save_model": None,        # *
            "epochs": 10,              # *

            "device": None,      
            
            "checkpoint_path": "./checkpoint/checkpoint", 
            "callbacks": [ModelRegistryCheckpoint()],
            "resume_from_checkpoint": None,
        },
    }

# the input size and the resize should match
if config["model_config"]['input_size'] != config["dataset_config"]['transform_config']['resize']:
    print("[warning] Input size and resize dimensions must match.")
    config["model_config"]['input_size'] = config["dataset_config"]['transform_config']['resize']
    print(f"Resizing images to: {config['dataset_config']['transform_config']['resize']}")

task.connect(config)

if config["trainer_config"]['load_model'] == "False" or config["trainer_config"]['load_model'] == "false" or config["trainer_config"]['load_model'] == "":
    config["trainer_config"]['load_model'] = None 
    
print(config)

model_reg = config["model_name"]

if config["trainer_config"]["save_model"] is not None:
    config["trainer_config"]["save_model"] = "model/"

# --------------     to load model -----------------

if config["trainer_config"]["load_model"] is not None : 
    model_id = manager.get_model_id_by_name(model_reg)

    manager.get_model(
        model_name= model_reg,  # or any valid model ID
        local_dest="."
    )
    # !!! ask from the data team to load from the right path
    config["trainer_config"]["load_model"] = f"./{model_id}/"

if config['trainer_config']["resume_from_checkpoint"] is not None:

    task_id = config['trainer_config']["resume_from_checkpoint"]

    checkpoint_name = f"checkpoint-{task_id}"
    print(f"Resuming from task ID: {task_id}")

    model_id = manager.get_model_id_by_name(checkpoint_name)
    manager.get_model(
        model_name=checkpoint_name,  # or any valid model ID
        local_dest="."
    )

    config["trainer_config"]["resume_from_checkpoint"] = f'./{model_id}/checkpoint'
        

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


# absolute_path = Path(__file__).parent / "dataset" / config["dataset_config"]["source"] / "images"

# config["dataset_config"]["source"] = str(absolute_path.resolve())

trainer = AutoTrainer(config=config)

trainer.run()

if config["trainer_config"]["save_model"] is not None:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name = model_reg + "_" + str(int(time.time())),
    )