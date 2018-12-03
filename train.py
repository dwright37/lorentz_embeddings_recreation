import numpy as np
import timeit
from torch.utils.data import DataLoader
import gc
import torch as th
th.set_default_tensor_type(th.DoubleTensor)
import logging
import argparse
from torch.autograd import Variable
from collections import defaultdict as ddict
import torch.multiprocessing as mp
import model, optimization
from data import DatasetReader
from optimization import RiemannianSGD
from sklearn.metrics import average_precision_score
import sys

_lr_multiplier = 0.1

def train_mp(model, data, optimizer, args, log, rank, queue):
    try:
        train(model, data, optimizer, args, log, rank, queue)
    except Exception as err:
        log.exception(err)
        queue.put(None)


def train(model, data, optimizer, args, log, rank=1, queue=None):
    # setup parallel data loader
    loader = DataLoader(
        data,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.ndproc,
        collate_fn=data.collate
    )

    for epoch in range(args.epochs):
        epoch_loss = []
        loss = None
        data.burnin = False
        lr = args.lr
        t_start = timeit.default_timer()
        if epoch < args.burnin:
            data.burnin = True
            lr = args.lr * _lr_multiplier
            if rank == 1:
                log.info('Burnin: lr=%.4f'%(lr))
        for inputs, targets in loader:
            inputs = Variable(th.from_numpy(np.vstack(inputs)))
            targets = Variable(th.from_numpy(np.vstack(targets))).squeeze()            

            elapsed = timeit.default_timer() - t_start
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss.append(loss.data[0])
            #th.save({
            #    'model': model.state_dict(),
            #    'epoch': epoch,
            #    'entities': data.entities
            #}, 'run2/%05d.pth'%epoch)
        if rank == 1:
            emb = None
            if epoch == (args.epochs - 1) or epoch % args.eval_each == (args.eval_each - 1):
                emb = model
            if queue is not None:
                queue.put(
                    (epoch, elapsed, np.mean(epoch_loss), emb)
                )
            else:
                log.info(
                    'info: {'
                    '"elapsed": %d, '%(elapsed),
                    '"loss": %f, '%(np.mean(epoch_loss)),
                    '}'
                )
        gc.collect()

def evaluate(types, model, distfn):
    with th.no_grad():
        lt = th.from_numpy(model.embedding())
        embedding = Variable(lt)
        ranks = []
        ap_scores = []
        for s, s_types in types.items():
            s_e = Variable(lt[s].expand_as(embedding))
            _dists = model.dist()(s_e, embedding).data.cpu().numpy().flatten()
            _dists[s] = 1e+12
            _labels = np.zeros(embedding.size(0))
            _dists_masked = _dists.copy()
            _ranks = []
            for o in s_types:
                _dists_masked[o] = np.Inf
                _labels[o] = 1
            ap_scores.append(average_precision_score(_labels, -_dists))
            for o in s_types:
                d = _dists_masked.copy()
                d[o] = _dists[o]
                r = np.argsort(d)
                _ranks.append(np.where(r == o)[0][0] + 1)
            ranks += _ranks
    return np.mean(ranks), np.mean(ap_scores)


def control(queue, log, types, data, fout, distfn, nepochs, processes):
    min_rank = (np.Inf, -1)
    max_map = (0, -1)
    while True:
        gc.collect()
        msg = queue.get()
        if msg is None:
            for p in processes:
                p.terminate()
            break
        else:
            epoch, elapsed, loss, model = msg
        if model is not None:
            # save model to fout
            th.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'entities': data.entities
            }, fout)
            # compute embedding quality
            mrank, mAP = evaluate(types, model, distfn)
            if mrank < min_rank[0]:
                min_rank = (mrank, epoch)
            if mAP > max_map[0]:
                max_map = (mAP, epoch)
            log.info(
                ('eval: {'
                 '"epoch": %d, '
                 '"elapsed": %.2f, '
                 '"loss": %.3f, '
                 '"mean_rank": %.2f, '
                 '"mAP": %.4f, '
                 '"best_rank": %.2f, '
                 '"best_mAP": %.4f}') % (
                     epoch, elapsed, loss, mrank, mAP, min_rank[0], max_map[0])
            )
        else:
            log.info('train: {{"epoch": {%d}, "loss": {%f}, "elapsed": {%d}}}'%(epoch, loss, elapsed))
        if epoch >= nepochs - 1:
            log.info(
                ('results: {'
                 '"mAP": %g, '
                 '"mAP epoch": %d, '
                 '"mean rank": %g, '
                 '"mean rank epoch": %d'
                 '}') % (
                     max_map[0], max_map[1], min_rank[0], min_rank[1])
            )
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Lorentz Embeddings')
    parser.add_argument('--dim', help='Embedding dimension', type=int)
    parser.add_argument('--dset', help='Dataset to embed', type=str)
    parser.add_argument('--fout', help='Filename where to store model', type=str)
    parser.add_argument('--lr', help='Learning rate', type=float)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('--batchsize', help='Batchsize', type=int, default=50)
    parser.add_argument('--negs', help='Number of negatives', type=int, default=20)
    parser.add_argument('--nproc', help='Number of processes', type=int, default=5)
    parser.add_argument('--ndproc', help='Number of data loading processes', type=int, default=2)
    parser.add_argument('--eval_each', help='Run evaluation each n-th epoch', type=int, default=10)
    parser.add_argument('--burnin', help='Duration of burn in', type=int, default=20)
    parser.add_argument('--debug', help='Print debug output', action='store_true', default=False)
    args = parser.parse_args()

    th.set_default_tensor_type('torch.DoubleTensor')
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    log = logging.getLogger('lorentz')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)
    data = DatasetReader(args.dset, args.negs)
    idx = data.samples

    # create adjacency list for evaluation
    adjacency = ddict(set)
    for i in range(len(idx)):
        s, o, _ = idx[i]
        adjacency[s].add(o)
    adjacency = dict(adjacency)

    # setup Riemannian gradients for distances
    distfn = optimization.LorentzDistance

    # initialize model and data
    model = model.LorentzEmbedding(len(data.entities), args.dim, distfn)

    # Build config string for log
    config = [
        ('dim', '{:d}'),
        ('lr', '{:g}'),
        ('batchsize', '{:d}'),
        ('negs', '{:d}'),
    ]
    config = ', '.join(['"{}": {}'.format(k, f).format(getattr(args, k)) for k, f in config])
    log.info('config: {{{}}}'.format(config))

    # initialize optimizer
    optimizer = RiemannianSGD(
        model.parameters(),
        lr=args.lr,
    )

    # if nproc == 0, run single threaded, otherwise run Hogwild
    if args.nproc == 0:
        train(model, data, optimizer, args, log, 0)
    else:
        queue = mp.Manager().Queue()
        model.share_memory()
        processes = []
        for rank in range(args.nproc):
            p = mp.Process(
                target=train_mp,
                args=(model, data, optimizer, args, log, rank + 1, queue)
            )
            p.start()
            processes.append(p)

        ctrl = mp.Process(
            target=control,
            args=(queue, log, adjacency, data, args.fout, distfn, args.epochs, processes)
        )
        ctrl.start()
        ctrl.join()
