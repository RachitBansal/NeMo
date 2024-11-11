# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for pretraining Mixture of Experts (MoE) models using NeMo and NeMo-Run.

This script provides functionality to configure and execute MoE model pretraining
on both local and Slurm-based systems. It leverages NeMo 2.0 recipes and NeMo-Run
for configuration and execution.

References:
    - NeMo: https://github.com/NVIDIA/NeMo
    - NeMo-Run: https://github.com/NVIDIA/NeMo-Run
"""

import argparse
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nemo_run as run
from nemo.collections import llm


def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser for MoE pretraining.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Custom MoE Pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_config",
        type=str,
        default="MoEConfig8x3B",
        help="MoE config from nemo.collections.llm.gpt.model.custom_moe"
    )
    model_group.add_argument(
        "--num_moe_experts",
        type=int,
        default=8,
        help="Number of MoE experts"
    )
    model_group.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of transformer layers"
    )
    model_group.add_argument(
        "--hidden_size",
        type=int,
        default=2048,
        help="Size of transformer hidden layers"
    )
    model_group.add_argument(
        "--num_attention_heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    model_group.add_argument(
        "--ffn_hidden_size",
        type=int,
        default=8192,
        help="Size of FFN hidden layers"
    )
    model_group.add_argument(
        "--max_position_embeddings",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    model_group.add_argument(
        "--seq_length",
        type=int,
        default=1024,
        help="Training sequence length"
    )

    # Dataset configuration  
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default="sample",
        help="Name of the dataset to use"
    )
    data_group.add_argument(
        "--tokenizer_type",
        type=str,
        default="GPT2BPETokenizer",
        help="Type of tokenizer to use"
    )

    # Slurm configuration
    slurm_group = parser.add_argument_group("Slurm Configuration") 
    slurm_group.add_argument(
        "--slurm",
        action="store_true",
        help="Run on Slurm using run.SlurmExecutor"
    )
    slurm_group.add_argument(
        "--slurm_host",
        type=str,
        default="holygpu8a19105",
        help="Slurm host node"
    )
    slurm_group.add_argument(
        "--slurm_user",
        type=str,
        default="brachit",
        help="Slurm username"
    )
    slurm_group.add_argument(
        "--slurm_account",
        type=str,
        default="kempner_grads",
        help="Slurm account name"
    )
    slurm_group.add_argument(
        "--slurm_partition",
        type=str,
        default="kempner_h100",
        help="Slurm partition name"
    )

    # Runtime configuration
    runtime_group = parser.add_argument_group("Runtime Configuration") 
    runtime_group.add_argument(
        "--num_nodes",
        type=int,
        default=2,
        help="Number of compute nodes"
    )
    runtime_group.add_argument(
        "--num_gpus_per_node", 
        type=int,
        default=4,
        help="Number of GPUs per node"
    )
    runtime_group.add_argument(
        "--dryrun",
        action="store_true",
        help="Perform a dry run without actual execution"
    )
    runtime_group.add_argument(
        "--base_output_dir",
        type=str,
        default="/n/holyscratch01/dam_lab/brachit/moes/optim/out",
        help="Base directory for output files"
    )

    # MoE and training configuration
    training_group = parser.add_argument_group("MoE and Training Configuration")
    training_group.add_argument(
        "--global_batch_size",
        type=int,
        default=64,
        help="Global batch size"
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    training_group.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Maximum number of training steps"
    )

    # Parallelism configuration
    parallel_group = parser.add_argument_group("Parallelism Configuration")
    parallel_group.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        default=1,
        help="Degree of tensor model parallelism"
    )
    parallel_group.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        default=1,
        help="Degree of pipeline model parallelism"
    )
    parallel_group.add_argument(
        "--virtual_pipeline_model_parallel_size",
        type=int,
        default=None,
        help="Number of pipeline stages for interleaved schedule"
    )
    parallel_group.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Degree of context parallelism"
    )
    parallel_group.add_argument(
        "--sequence_parallel",
        type=bool,
        default=False,
        help="Enable sequence parallelism"
    )
    parallel_group.add_argument(
        "--expert_model_parallel_size",
        type=int,
        default=8,
        help="Degree of expert model parallelism"
    )

    return parser


def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    time: str = "01:00:00",
    custom_mounts: Optional[List[str]] = None,
    custom_env_vars: Optional[Dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
) -> run.SlurmExecutor:
    """
    Create and configure a Slurm executor for distributed training.

    Args:
        user: Slurm username
        host: Slurm host address
        remote_job_dir: Remote directory for job files
        account: Slurm account name
        partition: Slurm partition name
        nodes: Number of nodes to use
        devices: Number of devices per node
        time: Job time limit
        custom_mounts: Additional mount points
        custom_env_vars: Additional environment variables
        container_image: Container image to use
        retries: Number of retry attempts

    Returns:
        run.SlurmExecutor: Configured Slurm executor
    """
    if not all([user, host, remote_job_dir, account, partition, nodes, devices]):
        raise RuntimeError(
            "Missing required Slurm configuration parameters. Please provide all required arguments."
        )

    mounts = custom_mounts or []

    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_TLS": "tcp",
    }
    if custom_env_vars:
        env_vars.update(custom_env_vars)

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir,
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres=f"gpu:{devices}",
        time="48:00:00",
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time
    executor.SRUN_ARGS = [
        "account", "partition", "job-name", "time", "nodes",
        "ntasks", "ntasks-per-node", "cpus-per-task",
        "gpus-per-node", "gpus-per-task", "qos", "mem",
        "mem-per-gpu", "mem-per-cpu", "comment", "constraint",
        "exclude", "gres", "exclusive", "array",
        "additional-parameters",
    ]

    return executor


def local_executor() -> run.LocalExecutor:
    """
    Create and configure a local executor for single-machine training.

    Returns:
        run.LocalExecutor: Configured local executor
    """
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    return run.LocalExecutor(env_vars=env_vars)

def get_dataset_paths(dataset_name: str) -> List[str]:
    """
    Get paths to dataset files based on dataset name.

    Args:
        dataset_name: Name of the dataset

    Returns:
        List[str]: List of dataset file paths
    """
    dataset_dirs = {
        "sample": [
            "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data/sample_dolma_testing",
        ],
        "dolma": [
            "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data/dolma_processed/books",
            "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data/dolma_processed/c4",
            "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data/dolma_processed/cc_en_head",
            "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data/dolma_processed/cc_en_middle",
            "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data/dolma_processed/cc_en_tail",
            "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data/dolma_processed/s2_v3",
            "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data/dolma_processed/v3",
            "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data/dolma_processed/wiki",
        ],
    }[dataset_name]

    paths = []
    for dataset_dir in dataset_dirs:
        for file in os.scandir(dataset_dir):
            if file.name.endswith('.bin'):
                prefix = str(Path(file.path).with_suffix(''))
                if os.path.exists(prefix + '.idx'):
                    paths.append(prefix)

    return paths

def get_vocab_paths(tokenizer_type: str) -> Tuple[str, str]:
    """
    Get paths to vocabulary and merges files based on tokenizer type.

    Args:
        tokenizer_type: Type of tokenizer

    Returns:
        Tuple[str, str]: Paths to vocabulary and merges files

    Raises:
        ValueError: If tokenizer type is not supported
    """
    if "gpt2" in tokenizer_type.lower():
        vocab_prefix = "/n/holylfs06/LABS/kempner_shared/Everyone/containers/mlperf_benchmarking/nemo_dev_code/data_tools/gpt2"
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return f"{vocab_prefix}-vocab.json", f"{vocab_prefix}-merges.txt"


def main() -> None:
    """Main execution function for MoE pretraining."""
    print("Starting main execution")
    args = get_parser().parse_args()
    
    if "NEMORUN_HOME" not in os.environ:
        raise RuntimeWarning(
            "NEMORUN_HOME environment variable is not set. "
            "This may cause files to be saved in the wrong directory."
        )

    exp_name = args.model_config
    data_paths = get_dataset_paths(args.dataset_name)
    logging.info(f"Data paths: {data_paths}")

    vocab_path, merges_path = get_vocab_paths(args.tokenizer_type)
    logging.info(f"Vocab path: {vocab_path}")
    logging.info(f"Merges path: {merges_path}")

    name = (
        f"model={args.model_config}"
        f"_dataset={args.dataset_name}"
        f"_tokenizer={args.tokenizer_type}"
        f"_num_experts={args.num_moe_experts}"
        f"_num_layers={args.num_layers}"
        f"_hidden_size={args.hidden_size}"
        f"_num_attention_heads={args.num_attention_heads}"
        f"_ffn_hidden_size={args.ffn_hidden_size}"
        f"_max_position_embeddings={args.max_position_embeddings}"
        f"_seq_length={args.seq_length}"
    )

    # Configure pretraining recipe
    pretrain = llm.custom_moe.pretrain_recipe(
        name=name,
        dir=os.environ["NEMORUN_HOME"],
        tokenizer=args.tokenizer_type,
        data_path=data_paths,
        vocab_path=vocab_path,
        merges_path=merges_path,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
    )
    logging.info(f"Pretraining recipe: {pretrain}")

    # Configure training strategy
    pretrain.trainer.strategy.tensor_model_parallel_size = args.tensor_model_parallel_size
    pretrain.trainer.strategy.pipeline_dtype = None
    pretrain.trainer.strategy.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    pretrain.trainer.strategy.context_parallel_size = args.context_parallel_size
    pretrain.trainer.strategy.sequence_parallel = args.sequence_parallel
    pretrain.trainer.strategy.expert_model_parallel_size = args.expert_model_parallel_size
    pretrain.trainer.val_check_interval = 500

    # Configure logging
    pretrain.log.ckpt.save_top_k = 10
    # pretrain.log.ckpt.every_n_train_steps = 500

    # Configure training parameters
    pretrain.trainer.max_steps = args.max_steps
    pretrain.optim.config.lr = args.learning_rate

    # Create appropriate executor
    if args.slurm:
        executor = slurm_executor(
            user=args.slurm_user,
            host=args.slurm_host,
            remote_job_dir=args.base_output_dir,
            account=args.slurm_account,
            partition=args.slurm_partition,
            nodes=pretrain.trainer.num_nodes,
            devices=pretrain.trainer.devices,
        )
        logging.info(f"Slurm executor: {executor}")
    else:
        executor = local_executor()
        logging.info(f"Local executor: {executor}")

    # Create and run experiment
    with run.Experiment(f"{exp_name}{args.dataset_name}") as exp:
        exp.add(
            pretrain,
            executor=executor,
            name=exp_name,
            tail_logs=isinstance(executor, run.LocalExecutor),
        )

        if args.dryrun:
            exp.dryrun()
        else:
            exp.run(sequential=True, detach=True)


if __name__ == "__main__":
    main()
