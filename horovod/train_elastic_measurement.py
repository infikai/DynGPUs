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

# Training settings
parser = argparse.ArgumentParser(description='Elastic PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('/mydata/Data/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/mydata/Data/imagenet/val'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')

# Horoovd settings
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

# Elastic Horovod settings
parser.add_argument('--batches-per-commit', type=int, default=500,
                    help='number of batches processed before calling `state.commit()`; '
                         'commits prevent losing progress if an error occurs, but slow '
                         'down training.')
parser.add_argument('--batches-per-host-check', type=int, default=1,
                    help='number of batches processed before calling `state.check_host_updates()`; '
                         'this check is very fast compared to state.commit() (which calls this '
                         'as part of the commit process), but because still incurs some cost due '
                         'to broadcast, so we may not want to perform it every batch.')

parser.add_argument('--model', type=str, default='vit_l_32', choices=['resnet50', 'vit_l_32'],
                    help='Model architecture to train')
# Default settings from https://arxiv.org/abs/1706.02677.
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


def train(state):
    print(f'Train() been called in rank {hvd.rank()}')
    model.train()
    epoch = state.epoch
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    batch_offset = state.batch
    print(f'Epoch: {epoch}; Batch offset:{batch_offset}')
    start_init_loop = time.time()
    for idx, (data, target) in enumerate(train_loader):
        start_batch = time.time()
        if idx == 0:
            print(f'Init for loop time: {start_batch - start_init_loop}s')
        # Elastic Horovod: update the current batch index this epoch
        # and commit / check for host updates. Do not check hosts when
        # we commit as it would be redundant.
        state.batch = batch_idx = batch_offset + idx
        if args.batches_per_commit > 0 and \
                state.batch % args.batches_per_commit == 0:
            start = time.time()
            state.commit()
            end = time.time()
            print(f'state commited! took {end - start}s')
        elif args.batches_per_host_check > 0 and \
                state.batch % args.batches_per_host_check == 0:
            state.check_host_updates()

        adjust_learning_rate(epoch, batch_idx)

        start_move = time.time()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        end_move = time.time()
        # print(f'Move data to GPU time: {end_move - start_move}s')
        optimizer.zero_grad()
        # if args.sleep > 0 and idx == 0:
        #     print(f'Sleep {args.sleep}s')
        #     time.sleep(args.sleep)
        # Split data into sub-batches of size batch_size
        start_train = time.time()
        for i in range(0, len(data), args.batch_size):
            if hvd.rank() == 0 and idx == 0:
                print(f'Time: {time.time() - start_train}s')
                int_train = time.time()
            data_batch = data[i:i + args.batch_size]
            if hvd.rank() == 0 and idx == 0:
                print(f'Time: {time.time() - int_train}s')
                int_train = time.time()
            target_batch = target[i:i + args.batch_size]
            if hvd.rank() == 0 and idx == 0:    
                print(f'Time: {time.time() - int_train}s')
                int_train = time.time()
            output = model(data_batch)
            if hvd.rank() == 0 and idx == 0:    
                print(f'Time: {time.time() - int_train}s')
                int_train = time.time()
            train_accuracy.update(accuracy(output, target_batch))
            if hvd.rank() == 0 and idx == 0:
                print(f'Time: {time.time() - int_train}s')
                int_train = time.time()
            loss = F.cross_entropy(output, target_batch)
            if hvd.rank() == 0 and idx == 0:
                print(f'Time: {time.time() - int_train}s')
                int_train = time.time()
            train_loss.update(loss)
            if hvd.rank() == 0 and idx == 0:
                print(f'Time: {time.time() - int_train}s')
                int_train = time.time()
            # Average gradients among sub-batches
            loss.div_(math.ceil(float(len(data)) / args.batch_size))
            if hvd.rank() == 0 and idx == 0:    
                print(f'Time: {time.time() - int_train}s')
                int_train = time.time()
            if args.sleep > 0 and idx == 0:
                print(f'Sleep {args.sleep}s')
                time.sleep(args.sleep)
            loss.backward()
            if hvd.rank() == 0 and idx == 0:    
                print(f'Time: {time.time() - int_train}s')
        end_train = time.time()
        print(f'Local train time: {end_train - start_train}s')

        # Elastic Horovod: record which samples were processed this batch
        # so we do not reprocess them if a reset event occurs
        state.train_sampler.record_batch(idx, allreduce_batch_size)

        start_op = time.time()
        # Gradient is applied across all ranks
        optimizer.step()
        end_batch = time.time()

        if hvd.rank() == 0:
            if idx % 1 == 0:
                print(f'Epoch: [{epoch + 1}][{idx}/{len(train_loader)}]\t'
                          f'Loss {loss.item():.4f}\t')
                print(f'Batch time: {end_batch - start_batch}s; Allreduce time: {end_batch - start_op}s')


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
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
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
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


def end_epoch(state):
    state.epoch += 1
    state.batch = 0
    state.train_sampler.set_epoch(state.epoch)
    state.commit()


# Horovod: average metrics from distributed training.
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
        # save_checkpoint(state.epoch)
        end_epoch(state)


if __name__ == '__main__':
    start_args = time.time()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(f'Args Time: {time.time() - start_args}s')

    start_main = time.time()
    if args.batch_size is None:
        args.batch_size = 2 if args.model == 'regnet_y_128gf' else 64

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    start_hvdInit = time.time()
    hvd.init()
    print(f'hvd init Time: {time.time() - start_hvdInit}s')
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    start_Tdataset = time.time()
    train_dataset = \
        datasets.ImageFolder(args.train_dir,
                             transform=transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    print(f'Tdataset Time: {time.time() - start_Tdataset}s')
    # Elastic Horovod: use ElasticSampler to partition data among workers.
    start_Tsampler = time.time()
    train_sampler = hvd.elastic.ElasticSampler(train_dataset)
    print(f'Tsampler Time: {time.time() - start_Tsampler}s')

    start_Tloader = time.time()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=allreduce_batch_size,
        sampler=train_sampler,
        **kwargs)
    print(f'Tloader Time: {time.time() - start_Tsampler}s')

    val_dataset = \
        datasets.ImageFolder(args.val_dir,
                             transform=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    val_sampler = hvd.elastic.ElasticSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        **kwargs)

    # Set up standard ResNet-50 model.
    print(f"Initializing model: {args.model}")
    start_model_loading = time.time()
    model = models.resnet50(weights=None) if args.model == 'resnet50' else models.vit_l_32(weights=None)
    end_model_loading = time.time()
    print(f"took: {end_model_loading - start_model_loading}s")
    # model = models.resnet50()

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    start_optim = time.time()
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)
    print(f"Init optim took: {time.time() - start_optim}s")

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    resume_from_epoch = 0
    if hvd.rank() == 0:
        for try_epoch in range(args.epochs, 0, -1):
            if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
                resume_from_epoch = try_epoch
                break

        if resume_from_epoch > 0:
            filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    print('Creating State: ')
    start_state = time.time()
    state = hvd.elastic.TorchState(model=model,
                                   optimizer=optimizer,
                                   train_sampler=train_sampler,
                                   val_sampler=val_sampler,
                                   epoch=resume_from_epoch,
                                   batch=0)
    print(f"took: {time.time() - start_state}s")
    end_main = time.time()
    print(f'Main function took: {end_main - start_main}s')

    full_train(state)

# 
# 