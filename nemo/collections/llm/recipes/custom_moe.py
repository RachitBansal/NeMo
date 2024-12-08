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


from typing import Any, Callable, Optional
import logging

import nemo_run as run
import pytorch_lightning as pl
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from pytorch_lightning.callbacks.callback import Callback

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
from nemo.collections.llm.gpt.model.custom_moe import MoEConfig8x3B, MoEModel, MoEConfig8x3BRandom
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe
from nemo.collections.llm.recipes.log.default import default_log, default_resume, wandb_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing, distributed_fused_adam_with_cosine_annealing_for_moe
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "moe_8x7b"

CONFIG_NAME_TO_MODEL_CONFIG = {
    "MoEConfig8x3B": MoEConfig8x3B,
    "MoEConfig8x3BRandom": MoEConfig8x3BRandom,
}

@run.cli.factory(name=NAME)
def model(
    config_name: str,
    seq_length: int,
    tokenizer: Any,
    optim: Any,
) -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Mixtral 8x7B model configuration.

    Args:
        seq_length (int): Sequence length.
        tokenizer (Any): Tokenizer.
        optim (Any): Optimizer.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Mixtral 8x7B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=mixtral_8x7b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(
        MoEModel, config=run.Config(
            CONFIG_NAME_TO_MODEL_CONFIG[config_name],
            seq_length=seq_length,
        ),
        tokenizer=tokenizer,
        optim=optim,
    )


def trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 4,
    pipeline_parallelism_type: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = 8,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    expert_parallelism: int = 8,
    num_nodes: int = 8,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for Mixtral 8x7B model.

    This function sets up the distributed training strategy optimized for the Mixtral 8x7B model.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        expert_parallelism (int): Degree of expert parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        callbacks (Optional[list[run.Config[Callback]]]): List of callback configurations.

    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.

    Examples:
        CLI usage:
            $ nemo llm pretrain trainer=mixtral_8x7b ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=2, num_gpus_per_node=8)
            >>> print(trainer_config)
    """
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        expert_model_parallel_size=expert_parallelism,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        limit_test_batches=50,
        limit_val_batches=32,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=run.Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=2000,
    )

    return trainer


def get_tokenizer(vocab_path, merges_path, tokenizer_name="GPT2BPETokenizer") -> Any:
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

    return get_nmt_tokenizer(
        "megatron",
        tokenizer_name,
        vocab_file=vocab_path,
        merges_file=merges_path,
    )

@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    config_name: str = "moe_8x7b",
    tokenizer: str = "GPT2BPETokenizer",
    data_path: str = "",
    vocab_path: str = "",
    merges_path: str = "",
    name: str = "default",
    num_nodes: int = 8,
    num_gpus_per_node: int = 8,
    seq_length: int = 4096,
    global_batch_size: int = 512,
    micro_batch_size: int = 32,
    performance_mode: bool = False,
    fn: Callable = pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for Mixtral 8x7B model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        config_name (str): Name of the model configuration to use.
        tokenizer (str): Tokenizer name.
        data_path (str): Path to the training folder (used for datatrove dataset).
        vocab_path (str): Path to the vocabulary file.
        merges_path (str): Path to the merges file.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        seq_length (int): Sequence length.
        global_batch_size (int): Global batch size.
        micro_batch_size (int): Micro batch size.
        performance_mode (bool): If true, enables optimizations for maximum performance.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory mixtral_8x7b
            $ nemo llm pretrain --factory "mixtral_8x7b(num_nodes=8, name='my_mixtral_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="mixtral_8x7b_pretrain", num_nodes=8)
            >>> print(recipe)
    """
    print(f"Initializing trainer with num_nodes={num_nodes}, num_gpus_per_node={num_gpus_per_node}")
    trainer_cfg = trainer(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        callbacks=[run.Config(TimingCallback)],
    )
    print(f"Initializing pretrain recipe with trainer_cfg={trainer_cfg}")
    data_cfg = run.Config(
        PreTrainingDataModule,
        paths=data_path,
        # tokenizer=get_tokenizer(vocab_path, merges_path, tokenizer),
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        rampup_batch_size=None,
        num_workers=0,
        split='90,5,5',
    )
    # data_cfg = run.Config(MockDataModule, seq_length=seq_length, global_batch_size=global_batch_size, micro_batch_size=1)
    optim_cfg = distributed_fused_adam_with_cosine_annealing_for_moe(
        max_lr=3e-4, max_lr_moe=1e-4
    )
    model_cfg = model(
        config_name=config_name,
        seq_length=seq_length,
        tokenizer=data_cfg.tokenizer,
        optim=optim_cfg,
    )
    recipe = run.Partial(
        fn,
        model=model_cfg,
        trainer=trainer_cfg,
        data=data_cfg,
        log=default_log(
            dir=dir,
            name=name,
            wandb_logger=wandb_logger(project="moes_optimization", name=name),
        ),
        optim=optim_cfg,
        resume=default_resume(),
    )

    if performance_mode:
        recipe = pretrain_performance_optimizations(recipe)

    return recipe


def pretrain_performance_optimizations(recipe: run.Partial) -> run.Partial:
    """
    Create a performance-optimized pre-training recipe for Mixtral 8x7B model.

    This method enables performance optimizations that may not be suitable for all use cases.
    It builds upon the standard pre-training recipe and adds additional performance enhancements.

    Args:
        recipe (run.Partial): Base pre-train recipe to which performance optimizations will be added

    Returns:
        run.Partial: Partial configuration for performance-optimized pre-training.

    Note:
        Use this method with caution and only when you need maximum performance.
        It may not be suitable for all hardware configurations or use cases.
    """

    # 'overlap_param_gather_with_optimizer_step' and 'align_param_gather' params are set automatically
    # by MegatronCommOverlapCallback. They are added here for user's knowledge.
    # overlap_param_gather_with_optimizer_step- Overlap param all-gather of first bucket with optimizer step.
    # align_param_gather- If true, all PP stages launch param all-gathers simultaneously, else
    # each PP stage launches independently as needed.

    recipe.trainer.callbacks.extend(
        [
            run.Config(MegatronTokenDropCallback),
            run.Config(
                MegatronCommOverlapCallback,
                overlap_param_gather_with_optimizer_step=True,
                align_param_gather=True,
            ),
        ]
    )

    return recipe


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
) -> run.Partial:
    """
    Create a fine-tuning recipe for Mixtral 8x7B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning. Allowed values: 'lora', 'none'/None.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory mixtral_8x7b
            $ nemo llm finetune --factory "mixtral_8x7b(num_nodes=2, name='my_mixtral_finetune')"

        Python API usage:
            >>> recipe = finetune_recipe(name="mixtral_8x7b_finetune", num_nodes=2)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning.
    """
    recipe = default_finetune_recipe(model(), "mistralai/Mixtral-8x7B-v0.1", dir, name, num_nodes, num_gpus_per_node)
    # recipe.trainer.strategy.expert_model_parallel_size = 8
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.pipeline_model_parallel_size = 4
        recipe.trainer.strategy.virtual_pipeline_model_parallel_size = 8
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() == 'lora':
        recipe.peft = run.Config(LoRA, target_modules=['linear_qkv', 'linear_proj'], dim=32)
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")
    return recipe