import os
import warnings
import logging

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("accelerate.utils.other").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="sacred")


from accelerate import Accelerator
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


import sacred
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

sacred.SETTINGS["CAPTURE_MODE"] = "sys"


accelerator = Accelerator()

logger = logging.getLogger(__name__)
if not accelerator.is_main_process:
    logger.setLevel(logging.ERROR)

experiment = Experiment()
experiment.logger = logger
experiment.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
if accelerator.is_main_process:
    experiment.observers.append(MongoObserver())


@experiment.config
def config():
    path = "samples/generated"
    num_samples = None
    batch_size = 30
    seed = 42

    # Conditioning config
    dataset_name = "DrawBench"
    dataset_kwargs = {}
    condition_kwargs = {}

    # Generation config
    solver_name = "Solver"
    solver_kwargs = {}
    model_name = "DiffusionModel"
    model_kwargs = {}

    assert torch.cuda.device_count() > 0, "No GPU found"

    # Evaluation config
    metric_name = "FID"
    metric_kwargs = {}


from core.pipeline import assess

experiment.main(assess)

if __name__ == "__main__":
    try:
        experiment.run_commandline()
    finally:
        accelerator.end_training()
