import datetime
import contextlib

from accelerate import Accelerator
import torch

accelerator = Accelerator()

# Idle ranks wait on this while rank 0 runs a long serial section: a dataset
# conversion, a per-dimension VBench subprocess. A gloo (CPU) group with a long
# timeout keeps NCCL's 10-min collective watchdog from aborting the run.
gloo_group = (
    torch.distributed.new_group(backend="gloo", timeout=datetime.timedelta(hours=2))
    if torch.distributed.is_initialized()
    else None
)


def gloo_barrier() -> None:
    if gloo_group is not None:
        torch.distributed.barrier(group=gloo_group)


@contextlib.contextmanager
def gloo_main_process_first():
    """
    `accelerator.main_process_first()` over `gloo_group`, for a rank-0-first section
    slow enough that waiting on the default NCCL group would trip the watchdog.
    """
    if not accelerator.is_main_process:
        gloo_barrier()
    try:
        yield
    finally:
        if accelerator.is_main_process:
            gloo_barrier()
