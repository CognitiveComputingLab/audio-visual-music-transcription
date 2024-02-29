import torch
import argparse
from base_dataloader import BaseDataLoader, DataPrefetcher
from base_dataset import BaseDataSet
from base_model import YourModel  # Replace with the actual name of your model class
from base_trainer import BaseTrainer

def main():
    # Argument parser to get configuration file path
    parser = argparse.ArgumentParser(description='Your Training Script')
    parser.add_argument('--config', type=str, default='path/to/your/config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    # Load configuration from the file
    config = load_config(args.config)

    # Create an instance of your dataset
    train_dataset = BaseDataSet(root=config['data']['train_data_dir'], split='train', mean=config['data']['mean'], std=config['data']['std'], ...)
    val_dataset = BaseDataSet(root=config['data']['val_data_dir'], split='val', mean=config['data']['mean'], std=config['data']['std'], val=True, ...)

    # Create an instance of your dataloader
    train_loader = BaseDataLoader(dataset=train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = BaseDataLoader(dataset=val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    # Create an instance of your model
    model = YourModel()  # Replace with the actual name of your model class

    # Create an instance of your trainer
    trainer = BaseTrainer(model, loss=None, resume=None, config=config, train_loader=train_loader, val_loader=val_loader, train_logger=None)

    # Start training
    trainer.train()

if __name__ == '__main__':
    main()

