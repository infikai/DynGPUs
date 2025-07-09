import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
import time

# --- Added new argument for device control ---
parser = argparse.ArgumentParser(description='Elastic PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device-control-file', default='device.txt',
                    help='File to control training device (gpu/cpu).')
# --- END MODIFICATION ---

parser.add_argument('--train-dir', default=os.path.expanduser('/mydata/Data/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/mydata/Data/imagenet/val'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--batches-per-commit', type=int, default=500,
                    help='number of batches processed before calling `state.commit()`')
parser.add_argument('--batches-per-host-check', type=int, default=1,
                    help='number of batches processed before calling `state.check_host_updates()`')
parser.add_argument('--model', type=str, default='vit_l_32', choices=['resnet50', 'vit_l_32'],
                    help='Model architecture to train')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--sleep', type=int, default=0,
                    help='sleep time')

# --- NEW: Global variable to track the current device ---
current_device = None

# --- NEW: Helper function to move optimizer tensors to the correct device ---
def move_optimizer_state(optimizer, device):
    """Moves the state of the optimizer to the specified device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

# --- NEW: Function to check control file and switch device if needed ---
def check_and_update_device(model, optimizer):
    """
    Checks the control file and performs a complete, synchronized device switch.
    """
    global current_device
    
    # Logic to decide and broadcast the target device (unchanged)
    if hvd.rank() == 0:
        try:
            with open(args.device_control_file, 'r') as f:
                target_device_str = f.read().strip().lower()
        except FileNotFoundError:
            target_device_str = 'gpu' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    else:
        target_device_str = None
    target_device_str = hvd.broadcast_object(target_device_str, root_rank=0)

    if target_device_str == 'gpu' and not args.no_cuda and torch.cuda.is_available():
        target_device = f'cuda:{hvd.local_rank()}'
    else:
        target_device = 'cpu'

    # Perform the switch if needed
    if target_device != current_device:
        print(f"Rank {hvd.rank()}: Syncing device! Moving from {current_device} to {target_device}")
        
        is_gpu_to_cpu_move = 'cuda' in current_device and target_device == 'cpu'
        
        if is_gpu_to_cpu_move:
            # =========================================================================
            # == THE COMPLETE 4-STEP GPU CLEANUP PROCEDURE ==
            # =========================================================================
            
            # 1. Clear stale gradients from the GPU
            optimizer.zero_grad(set_to_none=True)
            
            # 2. Move model parameters to CPU
            model.to(target_device)
            
            # 3. Move optimizer state (e.g., momentum buffers) to CPU
            move_optimizer_state(optimizer, target_device)
            
            # 4. Empty the cache of all unused memory blocks
            torch.cuda.empty_cache()
            
            print(f"Rank {hvd.rank()}: GPU cleanup complete.")
            # =========================================================================
        else:
            # For CPU to GPU, the order is simpler
            model.to(target_device)
            move_optimizer_state(optimizer, target_device)
            
        # Update the global device tracker
        current_device = target_device


def train(state):
    model.train()
    epoch = state.epoch
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    batch_offset = state.batch
    with tqdm(total=len(train_loader), desc=f'Train Epoch #{epoch+1}', disable=not verbose) as t:
        for idx, (data, target) in enumerate(train_loader):
            # --- MODIFIED: Check for device change at the start of each batch ---
            if hvd.rank() == 0: # Only rank 0 should check the file
                check_and_update_device(model, optimizer)
            
            # Broadcast the new device so all ranks are in sync
            device_int = 1 if 'cuda' in current_device else 0
            device_int_tensor = torch.tensor(device_int)
            device_int_tensor = hvd.broadcast(device_int_tensor, root_rank=0, name="device_sync")

            # All ranks update their device based on rank 0's decision
            local_target_device = f'cuda:{hvd.local_rank()}' if device_int_tensor.item() == 1 else 'cpu'
            if local_target_device != current_device:
                 print(f"Rank {hvd.rank()}: Syncing device! Moving from {current_device} to {local_target_device}")
                 model.to(local_target_device)
                 move_optimizer_state(optimizer, local_target_device)
                 globals()['current_device'] = local_target_device

            state.batch = batch_idx = batch_offset + idx
            if args.batches_per_commit > 0 and state.batch % args.batches_per_commit == 0:
                state.commit()
            elif args.batches_per_host_check > 0 and state.batch % args.batches_per_host_check == 0:
                state.check_host_updates()

            adjust_learning_rate(epoch, batch_idx)

            # --- MODIFIED: Move data to the dynamically set device ---
            data, target = data.to(current_device), target.to(current_device)

            optimizer.zero_grad()
            if args.sleep > 0:
                time.sleep(args.sleep)

            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()

            state.train_sampler.record_batch(idx, allreduce_batch_size)
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(), 'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)


    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
    state.commit()


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                # --- MODIFIED: Move data to the dynamically set device ---
                data, target = data.to(current_device), target.to(current_device)
                output = model(data)
                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(), 'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def end_epoch(state):
    state.epoch += 1
    state.batch = 0
    state.train_sampler.set_epoch(state.epoch)
    state.commit()


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


@hvd.elastic.run
def full_train(state):
    while state.epoch < args.epochs:
        train(state)
        validate(state.epoch)
        end_epoch(state)


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if use_cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
        current_device = f'cuda:{hvd.local_rank()}'
    else:
        current_device = 'cpu'

    cudnn.benchmark = True
    verbose = 1 if hvd.rank() == 0 else 0
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    train_dataset = datasets.ImageFolder(args.train_dir, transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    train_sampler = hvd.elastic.ElasticSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=allreduce_batch_size, sampler=train_sampler, **kwargs)

    val_dataset = datasets.ImageFolder(args.val_dir, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    val_sampler = hvd.elastic.ElasticSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, sampler=val_sampler, **kwargs)

    model = models.resnet50(weights=None) if args.model == 'resnet50' else models.vit_l_32(weights=None)

    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1
    if use_cuda and args.use_adasum and hvd.nccl_built():
        lr_scaler = args.batches_per_allreduce * hvd.local_size()

    model.to(current_device)

    optimizer = optim.SGD(model.parameters(), lr=(args.base_lr * lr_scaler), momentum=args.momentum, weight_decay=args.wd)
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=compression, backward_passes_per_step=args.batches_per_allreduce, op=hvd.Adasum if args.use_adasum else hvd.Average, gradient_predivide_factor=args.gradient_predivide_factor)

    # --- NEW: Check the control file once at startup ---
    if hvd.rank() == 0:
        check_and_update_device(model, optimizer)

    resume_from_epoch = 0
    state = hvd.elastic.TorchState(model=model, optimizer=optimizer, train_sampler=train_sampler, val_sampler=val_sampler, epoch=resume_from_epoch, batch=0)

    full_train(state)