import os
import sys
import json
import csv
import random
import argparse
import torch
import dataloaders
import models
import inspect
import math
from datetime import datetime
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import sys
import time
import wandb
from wandb import AlertLevel


class Coreset_Sampling():
    def __init__(self):
        pass


    def get_instance(self, module, name, config, *args):
        # GET THE CORRESPONDING CLASS / FCT 
        return getattr(module, config[name]['type'])(*args, **config[name]['args'])


    def create_episodedir(self,cfg, episode):
        episode_dir = os.path.join(cfg['exp_dir'], "episode"+str(episode))
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        else:
            print("=============================")
            print("Episode directory already exists: {}. Reusing it may lead to loss of old data in the directory.".format(episode_dir))
            print("=============================")

        cfg['episode'] = episode
        cfg['episode_dir'] = episode_dir
        cfg['trainer']['save_dir'] = os.path.join(episode_dir,cfg['trainer']['original_save_dir'])
        cfg['trainer']['log_dir'] = os.path.join(episode_dir,cfg['trainer']['original_log_dir'])

        cfg['labeled_loader']['args']['load_from'] = os.path.join(episode_dir, "labeled.txt")
        cfg['unlabeled_loader']['args']['load_from'] = os.path.join(episode_dir, "unlabeled.txt")
        cfg['labeled_rep_loader']['args']['load_from'] = os.path.join(episode_dir, "labeled.txt")

        return cfg


    def train_model(self, args, config):
        train_logger = Logger()

        # DATA LOADERS
        labeled_loader = self.get_instance(dataloaders, 'labeled_loader', config)
        val_loader = self.get_instance(dataloaders, 'val_loader', config)
        test_loader = self.get_instance(dataloaders, 'test_loader', config)

        # MODEL
        model = self.get_instance(models, 'arch', config, labeled_loader.dataset.num_classes)

        # LOSS
        loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

        # TRAINING
        trainer = Trainer(
            model=model,
            loss=loss,
            resume=args.resume,
            config=config,
            train_loader=labeled_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_logger=train_logger)

        trainer.train()

        config['checkpoint_dir'] = trainer._get_checkpoint_dir()
        config_save_path = os.path.join(config['checkpoint_dir'], 'updated_config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=True)


        return config

    def gpu_compute_dists(self, M1,M2):
            """
            Computes L2 norm square on gpu
            Assume 
            M1: M x D matrix
            M2: N x D matrix

            output: M x N matrix
            """
            #print(M1.shape)
            #print(M2.shape)
            #print(f"Function call to gpu_compute dists; M1: {M1.shape} and M2: {M2.shape}")
            M1_norm = (M1**2).sum(1).reshape(-1,1)

            M2_t = torch.transpose(M2, 0, 1)
            M2_norm = (M2**2).sum(1).reshape(1,-1)
            dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
            return dists



    def greedy_k_center(self, args, labeled, unlabeled):
            greedy_indices = [None for i in range(args.batch_size)]
            greedy_indices_counter = 0
            #move cpu to gpu
            labeled = torch.from_numpy(labeled)
            unlabeled = torch.from_numpy(unlabeled)

            print(f"[GPU] Labeled.shape: {labeled.shape}")
            print(f"[GPU] Unlabeled.shape: {unlabeled.shape}")
            # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
            st = time.time()
            min_dist,_ = torch.min(self.gpu_compute_dists(labeled[0,:].reshape((1,labeled.shape[1])), unlabeled), dim=0)
            min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))
            print(f"time taken: {time.time() - st} seconds")

            temp_range = 500
            dist = np.empty((temp_range, unlabeled.shape[0]))
            for j in tqdm(range(1, labeled.shape[0], temp_range), desc="Getting first farthest index"):
                if j + temp_range < labeled.shape[0]:
                    dist = self.gpu_compute_dists(labeled[j:j+temp_range, :], unlabeled)
                else:
                    dist = self.gpu_compute_dists(labeled[j:, :], unlabeled)

                min_dist = torch.cat((min_dist, torch.min(dist,dim=0)[0].reshape((1,min_dist.shape[1]))))

                min_dist = torch.min(min_dist, dim=0)[0]
                min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

            # iteratively insert the farthest index and recalculate the minimum distances:
            _, farthest = torch.max(min_dist, dim=1)
            greedy_indices [greedy_indices_counter] = farthest.item()
            greedy_indices_counter += 1

            amount = args.batch_size-1

            for i in tqdm(range(amount), desc = "Constructing Active set"):
                dist = self.gpu_compute_dists(unlabeled[greedy_indices[greedy_indices_counter-1], :].reshape((1,unlabeled.shape[1])), unlabeled)

                min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))

                min_dist, _ = torch.min(min_dist, dim=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))
                _, farthest = torch.max(min_dist, dim=1)
                greedy_indices [greedy_indices_counter] = farthest.item()
                greedy_indices_counter += 1

            remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
            remainSet = np.array(list(remainSet))

            return greedy_indices, remainSet



    def update_pools(self, args, config, episode):
        unlabeled_loader = self.get_instance(dataloaders, 'unlabeled_loader', config)
        labeled_rep_loader = self.get_instance(dataloaders, 'labeled_rep_loader', config)

        unlabeled_file = os.path.join(config["episode_dir"],"unlabeled.txt")
        unlabeled_reader = csv.reader(open(unlabeled_file, 'rt'))
        unlabeled_image_set = [r[0] for r in unlabeled_reader]

        # Model
        model = self.get_instance(models, 'arch', config, unlabeled_loader.dataset.num_classes)

        availble_gpus = list(range(torch.cuda.device_count()))
        device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

        # Load checkpoint
        #checkpoint = torch.load(os.path.join(config['checkpoint_dir'], "best_model.pth"), map_location=device)
        #if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        #    checkpoint = checkpoint['state_dict']
        # If during training, we used data parallel
        if not isinstance(model, torch.nn.DataParallel):
            # for gpu inference, use data parallel
            if "cuda" in device.type:
                model = torch.nn.DataParallel(model)
            else:
                # for cpu inference, remove module
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
                checkpoint = new_state_dict

        model.to(device)
        model.eval()

        unlabeled_features = []
        tbar = tqdm(unlabeled_loader, ncols=130)
        with torch.no_grad():
            for img_idx, (data, target) in enumerate(tbar):
                data, target = data.to(device), target.to(device)
                output, feature = model(data)
                output = output.cpu()
                feature = feature.cpu().numpy()
                unlabeled_features.append(feature)
        if len(unlabeled_features) > 1:
            unlabeled_features = np.concatenate(unlabeled_features, axis=0)
        else:
            unlabeled_features = unlabeled_features[0]

        labeled_features = []
        tbar = tqdm(labeled_rep_loader, ncols=130)
        with torch.no_grad():
            for img_idx, (data, target) in enumerate(tbar):
                data, target = data.to(device), target.to(device)
                output, feature = model(data)
                output = output.cpu()
                feature = feature.cpu().numpy()
                labeled_features.append(feature)
        if len(unlabeled_features) > 1:
            labeled_features = np.concatenate(labeled_features, axis=0)
        else:
            labeled_features = labeled_features[0]

        #print(labeled_features.shape)
        #print(unlabeled_features.shape)

        greedy_indexes, remainSet = self.greedy_k_center(args, labeled=labeled_features.reshape((labeled_features.shape[0], -1)), unlabeled=unlabeled_features.reshape(unlabeled_features.shape[0], -1))
        new_batch = []
        for index in greedy_indexes:
            new_batch.append(unlabeled_image_set[index])


        labeled = os.path.join(config['episode_dir'],"labeled.txt")
        labeled_reader = csv.reader(open(labeled, 'rt'))
        labeled_image_set = [r[0] for r in labeled_reader]

        new_labeled = labeled_image_set + new_batch
        new_labeled.sort()

        new_unlabeled = list(set(unlabeled_image_set) - set(new_batch))
        new_unlabeled.sort()

        config = self.create_episodedir(config, episode+1)

        with open(os.path.join(config['episode_dir'], "labeled.txt"), 'w') as f:
            writer = csv.writer(f)
            for image in new_labeled:
                writer.writerow([image])

        with open(os.path.join(config['episode_dir'], "unlabeled.txt"), 'w') as f:
            writer = csv.writer(f)
            for image in new_unlabeled:
                writer.writerow([image])

        return config
            


