import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
import os
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import ShapeMetrics, GraspStabilityMetrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from utils.vtg3d_utils import testModel, get_shuffled_subset, sample_visual_inputs
from tqdm import tqdm

class EmptyModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        return None

def run_net_3dvtg(args, config, train_writer=None, val_writer=None, test_writer=None):
    if args.vtg3d_shape_completion and args.vtg3d_grasp_stability:
        raise NotImplementedError('3D-VTG only support one task at a time')

    logger = get_logger(args.log_name)

    # build dataset
    (full_train_sampler, full_train_dataloader), (_, full_test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    if args.vtg3d_grasp_stability:
    # ablation config control
        for key, val in config.ablation.items():
            if key == 'direct_concat': # model-phase change
                config.grasp_model.direct_concat = val
            elif key == 'full_shape_input': # train-phase change
                pass
            elif key == 'without_spatial_align': # dataset-phase change
                config.dataset.train.others.without_spatial_align = val
                config.dataset.val.others.without_spatial_align = val
                config.dataset.test.others.without_spatial_align = val
            elif key == 'without_shape_completion': # train-phase and model-phase change
                config.grasp_model.without_shape_completion = val
            else:
                raise NotImplementedError(f'Unset ablation key: {key}')
        # addtional model config
        # config.grasp_model.loss_manager.total_epochs = config.max_epoch

    # build model
    if args.vtg3d_shape_completion:
        shape_model:nn.Module = builder.model_builder(config.shape_model)
        grasp_model:nn.Module = EmptyModule()
    elif args.vtg3d_grasp_stability:
        shape_model:nn.Module = builder.model_builder(config.shape_model)
        grasp_model:nn.Module = builder.model_builder(config.grasp_model)
        shape_ckpts = os.path.join(os.getcwd(), "ckpts", "ap_ps55.pth")
        builder.load_model(shape_model, shape_ckpts, logger = logger)
        # freeze all parameter in shape_model
        for param in shape_model.parameters():
            param.requires_grad = False

        # testModel(shape_model, grasp_model, config)
    else:
        raise NotImplementedError('3D-VTG only support grasp stability and shape completion task')

    if args.use_gpu:
        shape_model.to(args.local_rank)
        grasp_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        if args.vtg3d_shape_completion:
            start_epoch, best_metrics = builder.resume_model(shape_model, args, logger = logger)
            best_metrics = ShapeMetrics(config.consider_metric, best_metrics)
        else: 
            start_epoch, best_metrics = builder.resume_model(grasp_model, args, logger = logger)
            best_metrics = GraspStabilityMetrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        if args.vtg3d_shape_completion:
            builder.load_model(shape_model, args.start_ckpts, logger = logger)
        else:
            builder.load_model(grasp_model, args.start_ckpts, logger = logger)

    # print model info
    def printModelInfo(model, logger):
        print_log('=' * 10 + f'type{model}' + '=' * 10, logger = logger)
        print_log('Trainable_parameters:', logger = logger)
        print_log('=' * 25, logger = logger)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print_log(name, logger=logger)
        print_log('=' * 25, logger = logger)
        
        print_log('Untrainable_parameters:', logger = logger)
        print_log('=' * 25, logger = logger)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print_log(name, logger=logger)
        print_log('=' * 25, logger = logger)

    # printModelInfo(shape_model, logger = logger)
    # printModelInfo(grasp_model, logger = logger)



    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            shape_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(shape_model)
            grasp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(grasp_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        shape_model = nn.parallel.DistributedDataParallel(shape_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        grasp_model = nn.parallel.DistributedDataParallel(grasp_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        shape_model = nn.DataParallel(shape_model).cuda()
        grasp_model = nn.DataParallel(grasp_model).cuda()

    # optimizer & scheduler
    if args.vtg3d_shape_completion:
        optimizer = builder.build_optimizer(shape_model, config)
    elif args.vtg3d_grasp_stability:
        optimizer = builder.build_optimizer(grasp_model, config)
    else:
        raise NotImplementedError()

    # Criterion
    # shape completion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    # Grasp stability uses binary classification with BCEWithLogitsLoss.
    # CrossEntropyL1 = nn.CrossEntropyLoss()  # Previous multi-class formulation.
    BCELoss = nn.BCEWithLogitsLoss()  # For binary regression

    # Initialize AMP scaler for mixed precision training
    # scaler = GradScaler('cuda') if args.use_gpu else None
    scaler  = None
    # print_log('Using AMP (Automatic Mixed Precision) training...' if scaler else 'AMP disabled (CPU training)', logger=logger)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    if args.vtg3d_shape_completion:
        scheduler = builder.build_scheduler(shape_model, optimizer, config, last_epoch=start_epoch-1)
    elif args.vtg3d_grasp_stability:
        scheduler = builder.build_scheduler(grasp_model, optimizer, config, last_epoch=start_epoch-1)
    else:
        raise NotImplementedError()

    print("load ok")

    # trainval
    # training
    shape_model.zero_grad()
    grasp_model.zero_grad()

    use_vis_encoder = getattr(grasp_model.module.config, 'use_vis_encoder', True)
    
    train_dataloader, full_val_dataloader = get_shuffled_subset(full_train_dataloader.dataset, ratio=config.train_data_ratio, 
                                                            batch_size=config.total_bs, num_workers=args.num_workers,
                                                            max_sizes=(None, None), shuffle_btw_epoch=(True, False), shuffle_at_first=True, random_seed=42)
    val_dataloader, _ = get_shuffled_subset(full_val_dataloader.dataset, ratio=1, batch_size=24, num_workers=2,
                                             max_sizes=(23800, None), shuffle_btw_epoch=(False, False), shuffle_at_first=True, random_seed=42)
    test_dataloader, _ = get_shuffled_subset(full_test_dataloader.dataset, ratio=0.25, batch_size=24, num_workers=2,
                                             max_sizes=(23800, None), shuffle_btw_epoch=(False, False), shuffle_at_first=True, random_seed=42)
    
    print(f"Train dataloader size: {len(train_dataloader)}")
    print(f"Test dataloader size: {len(test_dataloader)}")
    print(f"Val dataloader size: {len(val_dataloader)}")
    # debug
    # epoch = -1
    # if args.vtg3d_shape_completion:
    #     _ = validate_shape_completion(shape_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)
    # elif args.vtg3d_grasp_stability:
    #     _ = validate_grasp_stability(shape_model, grasp_model, "UnSeen",  test_dataloader, epoch, CrossEntropyL1, test_writer, args, config, logger=logger)
    #     _ = validate_grasp_stability(shape_model, grasp_model, "Seen", val_dataloader, epoch, CrossEntropyL1, val_writer, args, config, logger=logger)

    print(f"start_epoch = {start_epoch}, max_epoch = {config.max_epoch}")
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            full_train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        if args.vtg3d_shape_completion:
            losses = AverageMeter(['SparseLoss', 'DenseLoss'])
        elif args.vtg3d_grasp_stability:
            losses = AverageMeter(['BCELoss'])
        else:
            raise NotImplementedError()

        num_iter = 0

        n_batches = len(train_dataloader)

        grasp_model.train()
        if args.vtg3d_grasp_stability:
            shape_model.eval()
        else:
            shape_model.train()
        
        for idx, item in enumerate(tqdm(train_dataloader, desc="Training Progress", unit="batch")):
            # unpack data and prepare model input
            assert item is not None, "Data is None"
            sc_data, gs_data, zero_mean, sample = item
            sc_input, sc_gt = sc_data[0].cuda(), sc_data[1].cuda()
            gs_input, result_gs = gs_data[0].cuda(), gs_data[1].cuda()
            obj_ids = sample[0]
            if isinstance(obj_ids, list):
                obj_ids = torch.tensor([int(obj_id) for obj_id in obj_ids], dtype=torch.long, device='cuda')
            else:
                obj_ids = obj_ids.cuda()
            data_time.update(time.time() - batch_start_time)

            # ablation study: full shape input, regard gt as shape input
            if args.vtg3d_grasp_stability and config.ablation.full_shape_input:
                # Randomly sample points from sc_gt to match the number of points in sc_input
                if sc_gt.shape[1] > sc_input.shape[1]:
                    gt_rand_idx = torch.randperm(sc_gt.shape[1])[:sc_input.shape[1]]
                    sc_input = sc_gt[:, gt_rand_idx, :] 
                else:
                    sc_input = sc_gt

            num_iter += 1
            if args.vtg3d_shape_completion:
                ret = shape_model(sc_input)
            elif args.vtg3d_grasp_stability:
                if config.ablation.without_shape_completion:
                    ret = grasp_model(None, sc_input, gs_input, obj_ids) # not using shape_model
                else:
                    _, vis_f, vis_coor = shape_model(sc_input, return_latent=True)
                    ret = grasp_model(vis_f, vis_coor, gs_input, obj_ids)  # logits: (B, 2)
            else:
                raise NotImplementedError()
        
            # get loss
            if args.vtg3d_shape_completion:
                sparse_loss, dense_loss = shape_model.module.get_loss(ret, sc_gt, epoch) # use 'module' to locate the real model obj
                _loss = sparse_loss + dense_loss 
                _loss.backward()
            elif args.vtg3d_grasp_stability:
                if result_gs.dim() == 2 and result_gs.size(1) == 1:
                    result_gs = result_gs.squeeze(1)
                result_gs = result_gs.float()  # Use float targets for binary classification.
                pred = ret
                # Convert pred from [B, 1] to [B] for BCEWithLogitsLoss.
                pred = pred.squeeze(1) if pred.dim() == 2 and pred.size(1) == 1 else pred
                total_loss, loss_info = grasp_model.module.loss_manager.get_loss(
                    pred,
                    result_gs,
                    epoch,
                )  # predictions, targets, epoch
                total_loss.backward()
            else:
                raise NotImplementedError()
            
            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(shape_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                torch.nn.utils.clip_grad_norm_(grasp_model.parameters(), getattr(config, 'grasp_grad_norm_clip', 5), norm_type=2)
                optimizer.step()
                num_iter = 0
                shape_model.zero_grad()
                grasp_model.zero_grad()

            if args.distributed:
                if args.vtg3d_shape_completion:
                    sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                    dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                    losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
                elif args.vtg3d_grasp_stability:
                    # Not tested
                    total_loss = dist_utils.reduce_tensor(total_loss, args)
                    losses.update([loss_info['bce_loss']])
            else:
                if args.vtg3d_shape_completion:
                    losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
                elif args.vtg3d_grasp_stability:
                    losses.update([loss_info['bce_loss']])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx  # Global batch index.
            if train_writer is not None:
                if args.vtg3d_shape_completion:
                    train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                    train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
                elif args.vtg3d_grasp_stability:
                    train_writer.add_scalar('Loss/Batch/BCE', loss_info['bce_loss'], n_itr)
                    

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                if args.vtg3d_grasp_stability:
                    # result_gs is the ground truth tensor for grasp stability
                    pos_cnt = (result_gs == 1).sum().item()
                    neg_cnt = (result_gs == 0).sum().item()
                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f PosCnt = %d NegCnt = %d' %
                        (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                        ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr'], pos_cnt, neg_cnt), logger = logger)
                else:
                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                        (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                        ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()
                    
        # one epoch end
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            if args.vtg3d_shape_completion:
                train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
                train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
            elif args.vtg3d_grasp_stability:
                train_writer.add_scalar('Loss/Epoch/BCE', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0: # val_freq default to 1
            if args.vtg3d_shape_completion:
                metrics = validate_shape_completion(shape_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)
            elif args.vtg3d_grasp_stability:
                seen_metrics = validate_grasp_stability(shape_model, grasp_model, "Seen", val_dataloader, epoch, BCELoss, val_writer, args, config, logger=logger)
                metrics = validate_grasp_stability(shape_model, grasp_model, "Unseen", test_dataloader, epoch, BCELoss, test_writer, args, config, logger=logger)
                print_log('[Validation] EPOCH: %d Cur-F1 = %.3f Pre-Best-F1 = %.3f' % (epoch, metrics.F1, best_metrics.F1 if best_metrics is not None else 0), logger=logger)

            if metrics is not None and (best_metrics is None or metrics.better_than(best_metrics)):
                best_metrics = metrics
                if args.vtg3d_shape_completion:
                    builder.save_checkpoint(shape_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                elif args.vtg3d_grasp_stability:
                    builder.save_checkpoint(grasp_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        if args.vtg3d_shape_completion:
            builder.save_checkpoint(shape_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        elif args.vtg3d_grasp_stability:
            builder.save_checkpoint(grasp_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if (config.max_epoch - epoch) < 2:
            if args.vtg3d_shape_completion:
                builder.save_checkpoint(shape_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
            elif args.vtg3d_grasp_stability:
                builder.save_checkpoint(grasp_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None and val_writer is not None and test_writer is not None:
        train_writer.close()
        val_writer.close()
        test_writer.close()

def validate_shape_completion(base_model, val_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(ShapeMetrics.names())
    category_metrics = dict()
    n_samples = len(val_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, item in enumerate(tqdm(val_dataloader, desc="Validation Progress", unit="batch")):
            assert item is not None, "Data is None"
            sc_data, gs_data, zero_mean, sample = item
            sc_input, sc_gt = sc_data[0].cuda(), sc_data[1].cuda()
            gs_input, result_gs = gs_data[0].cuda(), gs_data[1].cuda()
            obj_ids = sample[0] # sample = ( [obj_id_0, obj_id_1, ...], [grasp_id_0, grasp_id_1, ...] )

            # with autocast('cuda', enabled=args.use_gpu):
            ret = base_model(sc_input)
            coarse_points = ret[0]
            dense_points = ret[-1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, sc_gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, sc_gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, sc_gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, sc_gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = ShapeMetrics.get(dense_points, sc_gt)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _id in obj_ids:
                if _id not in category_metrics:
                    category_metrics[_id] = AverageMeter(ShapeMetrics.names()) # a list of metrics names
                category_metrics[_id].update(_metrics)
        
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
                
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'OBJID\t'
    msg += 'Count\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    print_log(msg, logger=logger)

    for _obj_id in category_metrics:
        msg = ''
        msg += (_obj_id + '\t')
        msg += (str(category_metrics[_obj_id].count(0)) + '\t')
        for value in category_metrics[_obj_id].avg():
            msg += '%.3f \t' % value
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return ShapeMetrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net_3dvtg(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    if args.vtg3d_shape_completion:
        test_losses = AverageMeter(['SparseLoss', 'DenseLoss'])
    elif args.vtg3d_grasp_stability:
        test_losses = AverageMeter(['BCELoss'])
    else:
        raise NotImplementedError()

    if args.vtg3d_shape_completion:
        test_metrics = AverageMeter(ShapeMetrics.names())
    elif args.vtg3d_grasp_stability:
        raise NotImplementedError() #TODO
    else:
        raise NotImplementedError()
    
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    interval =  n_samples // 10

    with torch.no_grad():
        for idx, item in enumerate(tqdm(test_dataloader, desc="Testing Progress", unit="batch")):
            # unpack data and prepare model input
            assert item is not None, "Data is None"
            sc_data, gs_data, zero_mean, sample = item
            sc_input, sc_gt = sc_data[0].cuda(), sc_data[1].cuda()
            gs_input, result_gs = gs_data[0].cuda(), gs_data[1].cuda()
            obj_ids = sample[0]

            if args.vtg3d_shape_completion:
                # forward
                ret = base_model(sc_input)
                ret = ret.float()
                # unpack ret
                coarse_points = ret[0]
                dense_points = ret[-1]
                # get loss
                sparse_loss_l1 =  ChamferDisL1(coarse_points, sc_gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, sc_gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, sc_gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, sc_gt)
                # update loss
                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])
                _metrics = ShapeMetrics.get(dense_points, sc_gt, require_emd=True)

                for _id in obj_ids:
                    if _id not in category_metrics:
                        category_metrics[_id] = AverageMeter(ShapeMetrics.names()) # a list of metrics names
                    category_metrics[_id].update(_metrics)

            elif args.vtg3d_grasp_stability:
                ret = base_model(sc_input, gs_input)
                raise NotImplementedError()

            else:
                raise NotImplementedError()

            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)

        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

    # Print testing results 
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'OBJID\t'
    msg += 'Count\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    print_log(msg, logger=logger)

    for _obj_id in category_metrics:
        msg = ''
        msg += (_obj_id + '\t')
        msg += (str(category_metrics[_obj_id].count(0)) + '\t')
        for value in category_metrics[_obj_id].avg():
            msg += '%.3f \t' % value
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return

def validate_grasp_stability(shape_model, grasp_model, val_cls:str, val_cls_dataloader, epoch, criterion, val_cls_writer, args, config, logger=None):
    print_log(f"[{val_cls} VALIDATION] Start validating epoch {epoch} for Class {val_cls}", logger=logger)
    
    print(f"Validation dataloader size: {len(val_cls_dataloader)}")
    shape_model.eval()
    grasp_model.eval()

    val_losses = AverageMeter(['BCELoss'])
    # Accumulate all logits and gts per object
    object_logits = dict()
    object_gts = dict()
    n_samples = len(val_cls_dataloader)
    interval = n_samples // 10 if n_samples >= 10 else 1
    pos_cnt = 0
    neg_cnt = 0

    with torch.no_grad():
        for idx, item in enumerate(tqdm(val_cls_dataloader, desc="Validation Progress", unit="batch")):
            sc_data, gs_data, zero_mean, sample = item
            sc_input, sc_gt = sc_data[0].cuda(), sc_data[1].cuda()
            gs_input, result_gs = gs_data[0].cuda(), gs_data[1].cuda()
            obj_ids = sample[0]

            # ablation study: full shape input
            if args.vtg3d_grasp_stability and config.ablation.full_shape_input:
                # Randomly sample points from sc_gt to match the number of points in sc_input
                if sc_gt.shape[1] > sc_input.shape[1]:
                    gt_rand_idx = torch.randperm(sc_gt.shape[1])[:sc_input.shape[1]]
                    sc_input = sc_gt[:, gt_rand_idx, :] 
                else:
                    sc_input = sc_gt

            if config.ablation.without_shape_completion:
                logits = grasp_model(None, sc_input, gs_input)  # without shape completion
            else:
                _, vis_f, vis_coor = shape_model(sc_input, return_latent=True)
                logits = grasp_model(vis_f, vis_coor, gs_input)  # [B, 2]

            if result_gs.dim() == 2 and result_gs.size(1) == 1:
                result_gs = result_gs.squeeze(1)
            result_gs = result_gs.float()  # Use float targets for binary classification.

            batch_pos_cnt = (result_gs == 1).sum().item()
            batch_neg_cnt = (result_gs == 0).sum().item()
            pos_cnt += batch_pos_cnt
            neg_cnt += batch_neg_cnt

            logits = logits.float()
            # Squeeze [B, 1] outputs to [B].
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)

            loss = criterion(logits, result_gs)
            val_losses.update([loss.item()])

            # Accumulate logits and gts per object
            for i, _id in enumerate(obj_ids):
                if _id not in object_logits:
                    object_logits[_id] = []
                    object_gts[_id] = []
                object_logits[_id].append(logits[i].detach().cpu())
                object_gts[_id].append(result_gs[i].detach().cpu())

            if (idx + 1) % interval == 0:
                # Use sigmoid plus thresholding for binary classification.
                if logits.dim() == 1:  # [B] shape
                    choices = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                    gts = result_gs.cpu().numpy()
                    logit_str = str(logits[0].detach().cpu().numpy())
                    choice_str = str(int(choices[0]))
                    gt_str = str(int(gts[0]))
                else:  # [B, 1] shape (shouldn't happen after squeeze but just in case)
                    choices = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                    gts = result_gs.cpu().numpy()
                    logit_str = str(logits[0].detach().cpu().numpy())
                    choice_str = str(int(choices[0]))
                    gt_str = str(int(gts[0]))
                print_log(
                    'Test[%d/%d] Losses = %s | Logit = %s | Choice = %s | GT = %s' % (
                        idx + 1,
                        n_samples,
                        ['%.4f' % l for l in val_losses.val()],
                        logit_str,
                        choice_str,
                        gt_str
                    ),
                    logger=logger
                )
    # Print overall pos/neg counts
    print_log(f'[Validation] EPOCH: {epoch} PosCnt = {pos_cnt} NegCnt = {neg_cnt}', logger=logger)

    # Compute per-object metrics and accumulate in test_metrics
    category_metrics = dict()
    metric_names = GraspStabilityMetrics.names()
    test_metrics = AverageMeter(metric_names)
    for _obj_id in object_logits:
        logits_tensor = torch.stack(object_logits[_obj_id], dim=0)
        gts_tensor = torch.stack(object_gts[_obj_id], dim=0)
        metrics = GraspStabilityMetrics.get(logits_tensor, gts_tensor)
        metrics = [_metric.item() if hasattr(_metric, 'item') else float(_metric) for _metric in metrics]
        category_metrics[_obj_id] = metrics
        test_metrics.update(metrics)

    # Compute overall metrics using test_metrics
    overall_metrics = GraspStabilityMetrics(metric_names, test_metrics.avg())

    print_log('[Validation][%s][GraspStability] EPOCH: %d  Metrics = %s' %
              (val_cls, epoch, ['%.4f' % m for m in overall_metrics._values]), logger=logger)

    # Print testing results
    print_log('============================ VALIDATION RESULTS ============================', logger=logger)
    msg = 'ID\tCount\t'
    for metric in metric_names:
        msg += metric + '\t'
    print_log(msg, logger=logger)

    for _obj_id in category_metrics:
        msg = ''
        msg += (_obj_id + '\t')
        msg += (str(len(object_logits[_obj_id])).zfill(3) + '\t')
        for value in category_metrics[_obj_id]:
            msg += '%.3f \t' % value
        print_log(msg, logger=logger)

    msg = 'Overall\t'
    for value in overall_metrics._values:
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    if val_cls_writer is not None:
        val_cls_writer.add_scalar('Loss/Epoch/BCE', val_losses.avg(0), epoch)
        for i, metric in enumerate(metric_names):
            val_cls_writer.add_scalar(f'/Metric/{metric}', overall_metrics._values[i], epoch)

    return GraspStabilityMetrics(config.consider_metric, overall_metrics._values) if overall_metrics._values else None
