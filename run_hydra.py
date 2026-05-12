import torch
from src.train_distributed import train
from src.optim.utils import get_scheduler, get_optimizer_factory
from src.utils import set_seed, get_worker_info
from src.model import load_model
from src.dataloader import ShardedDataLoader, get_dataset_path
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import copy 
import json
import os
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import uuid


@hydra.main(version_base=None, config_path="hydra_conf")
def main(config : DictConfig):
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(42)

    # First set up DDP
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        dist.init_process_group(backend='nccl')
    world_size, rank, local_rank, device = get_worker_info()
    master_process = (rank == 0) # this process will do logging, checkpointing etc.
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"Using device: {device}")

    # Logging
    output_dir = HydraConfig.get().runtime.output_dir
    ckpt_dir = f"{output_dir}/checkpoints/" if config['logging_params'].get('save_checkpoint', False) else ""
    if master_process:
        print(f"Loading configuration from {HydraConfig.get().job.config_name}")
        print(f"Training on dataset {config['recipe']['dataset']['name']}")
        os.makedirs(output_dir, exist_ok=True)  
        if ckpt_dir != "": os.makedirs(ckpt_dir, exist_ok=True)

    # Load model
    model = load_model(config['recipe']['model'], device)
    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if master_process:
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    print(model)
    # Set the training parameters
    training_params = config['recipe']['training_params']
    opt_config = config["optimizer_params"]
    data_config = config['recipe']['dataset']
    torch.set_float32_matmul_precision(training_params['tensorcore_precision'])

    # Load data
    tokenizer_name = data_config.get('tokenizer', 'gpt2')
    dataset_path = get_dataset_path(data_config['name'], tokenizer_name)
    if master_process: print(f"Load data from {dataset_path}")
    B, T = training_params['batch_size'], training_params['context_length']
    assert training_params['tokens_processed'] % (world_size * B * T) == 0 
    assert config['logging_params']['val_tokens_processed'] % (world_size * B * T) == 0
    num_shards = None
    if 'total_tokens_B' in data_config:
        num_shards = int(data_config['total_tokens_B'] * 10)  # 0.1B per shard
        if master_process: print(f"Limiting training to {data_config['total_tokens_B']}B tokens ({num_shards} shards)")
    train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device, num_shards=num_shards)
    val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    if master_process:
        print(f"Length of train dataset : {len(train_dataloader)/1e6:0.1f} million tokens")
        print(f"Length of validation dataset : {len(val_dataloader)/1e6:0.1f} million tokens")
        print(f"Total number of iterations : {total_iterations}")

    if master_process:
        print(f"Training with optimizer {opt_config['name']} and learning rate {opt_config['args']['lr']}")

    # Generate hash for the current optimizer configuration
    random_job_id = str(uuid.uuid4()).split('-')[0]
    file_name = f"logs_jobid_{random_job_id}.json"
    output_path = os.path.join(output_dir, file_name)

    # copy model to ensure consistency
    model_copy = copy.deepcopy(model).to(device)
    if training_params['compile']:
        if master_process: print("Compiling model")
        model_copy = torch.compile(model_copy)

    if ddp:
        model_copy = DDP(model_copy, device_ids=[local_rank])

    # Setup optimizer
    optimizer_obj = get_optimizer_factory(opt_config['name'])
    opt_config_args = opt_config['args']
    if opt_config['name'] in ['muon', 'muonq']: opt_config_args['nheads'] = config['recipe']['model'].get('n_head', None)
    optimizer = optimizer_obj(model_copy.named_parameters(), **opt_config_args)
    scheduler = get_scheduler(config['lr_schedule'], optimizer, total_iterations=total_iterations)

    # Initialize wandb
    if master_process and config['logging_params'].get('wandb', None) is not None:
        config_for_wandb_logging = dict(**config, world_size=world_size)
        wandb_config = config['logging_params']['wandb']
        if "dir" not in wandb_config:
            wandb_config['dir'] = "./outputs/wandb"
        wandb_run = wandb.init(
            **wandb_config,
            id=random_job_id,
            tags=wandb_config.get('tags', []) + [str(HydraConfig.get().job.name)],
            config=config_for_wandb_logging,
            reinit='create_new',
        )
    else:
        wandb_run = None

    # Train
    try:
        logger = train(train_dataloader, val_dataloader, model_copy, optimizer, training_params,
                    scheduler=scheduler, ckpt_dir=ckpt_dir,
                    logging_params=config['logging_params'], wandb_run=wandb_run)
    finally:
        if master_process and wandb_run is not None:
            wandb_run.finish()

    # Save
    if master_process:
        logger.name = opt_config['name'] + '-lr-' + str(opt_config['args']['lr'])

        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Overwriting")
        with open(output_path, 'w') as file:
            json.dump(logger.__dict__, file)
        print(f"Saved output to {output_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
