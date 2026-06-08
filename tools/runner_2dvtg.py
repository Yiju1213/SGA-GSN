from sklearn import dummy
import torch
import torch.nn as nn
import os
import time
from tools import builder
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import GraspStabilityMetrics
from utils.vtg3d_utils import get_shuffled_subset
from tqdm import tqdm
from torch.utils.data import DataLoader

def testModel2D(grasp_model, config, num_warmup=10, num_runs=100):
    """
    Test 2D grasp model to measure FLOPs, Parameters, inference latency and throughput
    """
    import time
    from thop import profile
    
    # Define input dimensions for 2D CNN_MCA model
    B = 1  # Batch size
    C, H, W = 3, 224, 224  # RGB channels, height, width
    
    # Create dummy inputs
    visual_rgb = torch.randn(B, C, H, W)   # Visual RGB image (B, 3, 224, 224)
    tactile_rgb = torch.randn(B, C, H, W)  # Tactile RGB concat (B, 3, 224, 224) 
    
    # Move model and inputs to GPU
    grasp_model.to('cuda')
    grasp_model.eval()  # Set to eval mode for proper inference timing
    
    visual_rgb = visual_rgb.cuda()
    tactile_rgb = tactile_rgb.cuda()
    
    # Warmup runs to stabilize GPU
    print("Warming up 2D model...")
    with torch.no_grad():
        for _ in range(num_warmup):
            if config.grasp_model.NAME == 'GraspStability_CNNMCA':
                _ = grasp_model(visual_rgb, tactile_rgb)
            elif config.grasp_model.NAME == 'GraspStability_CNN':
                _ = grasp_model(visual_rgb, visual_rgb, 
                                tactile_rgb, tactile_rgb, 
                                tactile_rgb, tactile_rgb)
    
    # Measure inference latency
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            if config.grasp_model.NAME == 'GraspStability_CNNMCA':
                out = grasp_model(visual_rgb, tactile_rgb)
            elif config.grasp_model.NAME == 'GraspStability_CNN':
                out = grasp_model(visual_rgb, visual_rgb, 
                                  tactile_rgb, tactile_rgb, 
                                  tactile_rgb, tactile_rgb)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = total_time / num_runs
    throughput = num_runs / total_time  # samples per second
    
    print(f"=== 2D Model Performance Metrics ===")
    print(f"Average Inference Latency: {avg_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    # Profile the grasp model
    if config.grasp_model.NAME == 'GraspStability_CNNMCA':
        flops, params = profile(grasp_model, (visual_rgb, tactile_rgb))
    elif config.grasp_model.NAME == 'GraspStability_CNN':
        flops, params = profile(grasp_model, (visual_rgb, visual_rgb, 
                                              tactile_rgb, tactile_rgb, 
                                              tactile_rgb, tactile_rgb))
    print(f'2D Grasp Model {config.grasp_model.NAME} FLOPs: {flops/1e9:.2f}G')
    print(f'2D Grasp Model {config.grasp_model.NAME} Params: {params/1e6:.2f}M')
    
    # Forward pass to check output shape
    with torch.no_grad():
        if config.grasp_model.NAME == 'GraspStability_CNNMCA':
            out = grasp_model(visual_rgb, tactile_rgb)
        elif config.grasp_model.NAME == 'GraspStability_CNN':
            out = grasp_model(visual_rgb, visual_rgb, 
                              tactile_rgb, tactile_rgb, 
                              tactile_rgb, tactile_rgb)
        print(f"2D Grasp Model output shape: {out.shape}")  # Expected: (B, num_classes)
        
        # Check for NaN values
        if torch.isnan(out).any():
            print("Warning: NaN detected in 2D grasp model output!")
    
    # Return metrics for further analysis
    return {
        'avg_latency_ms': avg_latency * 1000,
        'throughput_sps': throughput,
        'flops': flops,
        'params': params,
        'output_shape': out.shape
    }

def run_net_2dvtg(args, config, train_writer=None, val_writer=None, test_writer=None):
    """
    Simplified 2D VTG runner for grasp stability prediction using CNN_MCA model
    Focus: single GPU, grasp stability task only, train & validation
    """
    logger = get_logger(args.log_name)
    
    # Build 2D CNN_MCA model for grasp stability
    grasp_model = builder.model_builder(config.grasp_model)
    print_log(f'Built 2D CNN_MCA model: {config.grasp_model.NAME}', logger=logger)
    
    if args.use_gpu:
        assert torch.cuda.is_available(), "CUDA is not available but args.use_gpu is set."
        grasp_model = grasp_model.cuda()
        print_log('Using single GPU training...', logger=logger)
    
    # Test model FLOPs and Parameters
    testModel2D(grasp_model, config)
    return
    
    # Build 2D dataset
    (full_train_sampler, full_train_dataloader), (_, full_test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                              builder.dataset_builder(args, config.dataset.val)
    
    # Create subset dataloaders following 3dvtg pattern
    train_dataloader, full_val_dataloader = get_shuffled_subset(full_train_dataloader.dataset, ratio=config.get('train_data_ratio', 0.75),
                                            batch_size=config.total_bs, num_workers=args.num_workers,
                                            max_sizes=(None, None), shuffle_btw_epoch=(True, False), 
                                            shuffle_at_first=True, random_seed=42)
    
    val_dataloader, _ = get_shuffled_subset(full_val_dataloader.dataset, ratio=0.25, 
                                          batch_size=config.get('val_bs', 200), num_workers=1,
                                          max_sizes=(23800, None), shuffle_btw_epoch=(False, False), 
                                          shuffle_at_first=True, random_seed=42)
    
    test_dataloader, _ = get_shuffled_subset(full_test_dataloader.dataset, ratio=0.25, 
                                           batch_size=config.get('test_bs', 200), num_workers=1,
                                           max_sizes=(23800, None), shuffle_btw_epoch=(False, False), 
                                           shuffle_at_first=True, random_seed=42)
    
    print(f"Train dataloader size: {len(train_dataloader)}")
    print(f"Val dataloader size: {len(val_dataloader)}")
    print(f"Test dataloader size: {len(test_dataloader)}")
    
    # Training parameters
    start_epoch = 0
    best_metrics = None
    
    # Resume checkpoint if specified
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(grasp_model, args, logger=logger)
        best_metrics = GraspStabilityMetrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(grasp_model, args.start_ckpts, logger=logger)
    
    # 
    grasp_model = nn.DataParallel(grasp_model).cuda() # to fit builder function
    
    # Optimizer & Scheduler
    optimizer = builder.build_optimizer(grasp_model, config)
    scheduler = builder.build_scheduler(grasp_model, optimizer, config, last_epoch=start_epoch-1)
    
    # Loss criterion for binary grasp stability
    criterion = nn.BCEWithLogitsLoss()
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    
    # Training loop
    print_log(f"Starting training from epoch {start_epoch} to {config.max_epoch}", logger=logger)
    
    # debug: validation on val/test
    # seen_metrics = validate_2d_grasp_stability(grasp_model, "Seen", val_dataloader, start_epoch, criterion, val_writer, args, config, logger=logger)
    # metrics = validate_2d_grasp_stability(grasp_model, "Unseen", test_dataloader, start_epoch, criterion, test_writer, args, config, logger=logger)
    
    for epoch in range(start_epoch, config.max_epoch + 1):
        grasp_model.train()
        
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['BCELoss'])
        
        n_batches = len(train_dataloader)
        
        for idx, item in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} Training", unit="batch")):
            data_time.update(time.time() - batch_start_time)
            
            if config.grasp_model.NAME == 'GraspStability_CNNMCA':
              # Unpack 2D data: visual and tactile images
              visual_rgb = item['visual_rgb'].cuda()  # [B, 3, 224, 224]
              tactile_rgb = item['tactile_rgb_concat'].cuda() # [B, 3, 224, 224]
              grasp_result = item['grasp_result'].cuda()  # [B] 
              obj_id, grasp_id = item['sample_info']  # [B, 2] - object and grasp IDs
              # Forward pass
              logits = grasp_model(visual_rgb, tactile_rgb)  # [B, num_classes]
            elif config.grasp_model.NAME == 'GraspStability_CNN':
              # Unpack 2D data for CNN model: visual RGB and separate tactile sensors
              visual_rgb = item['visual_rgb'].cuda()  # [B, 3, 224, 224]
              tactile_rgb_left = item['tactile_rgb_left'].cuda()  # [B, 3, 224, 224]
              tactile_rgb_right = item['tactile_rgb_right'].cuda()  # [B, 3, 224, 224]
              tactile_bg_left = item['tactile_bg_left'].cuda()  # [B, 3, 224, 224]
              tactile_bg_right = item['tactile_bg_right'].cuda()  # [B, 3, 224, 224]
              grasp_result = item['grasp_result'].cuda()  # [B]
              obj_id, grasp_id = item['sample_info']  # [B, 2] - object and grasp IDs
              
              # Map dataset outputs to CNN model inputs:
              # cam_bef = cam_dur = visual_rgb (no temporal separation for visual)
              # lgel_bef = tactile_bg_left, lgel_dur = tactile_rgb_left  
              # rgel_bef = tactile_bg_right, rgel_dur = tactile_rgb_right
              logits = grasp_model(visual_rgb, visual_rgb, 
                                 tactile_bg_left, tactile_rgb_left,
                                 tactile_bg_right, tactile_rgb_right)  # [B, num_classes]
            else:
              raise NotImplementedError(f"Model {config.grasp_model.NAME} is not supported yet in this runner.")
        
            
            # Compute loss for binary classification
            if len(logits.shape) == 1 or logits.size(1) == 1:  # Single output for binary classification
                logits = logits.squeeze() if logits.size(1) == 1 else logits  # [B]
            elif logits.size(1) == 2:  # Two class outputs, use positive class
                logits = logits[:, 1]  # [B] - positive class logits
            
            loss = criterion(logits, grasp_result)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(grasp_model.parameters(), getattr(config, 'grad_norm_clip', 5.0), norm_type=2)
            
            optimizer.step()
            
            # Update metrics
            losses.update([loss.item()])
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            # Logging
            if idx % 100 == 0:
                pos_cnt = (grasp_result == 1).sum().item()
                neg_cnt = (grasp_result == 0).sum().item()
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) BCELoss = %.4f lr = %.6f PosCnt = %d NegCnt = %d' %
                    (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                    losses.val()[0], optimizer.param_groups[0]['lr'], pos_cnt, neg_cnt), logger=logger)
            
            # TensorBoard logging
            if train_writer is not None:
                n_itr = epoch * n_batches + idx
                train_writer.add_scalar('Loss/Batch/BCE', loss.item(), n_itr)
        
        # End of epoch
        if isinstance(scheduler, list):
          for item in scheduler:
              item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()
        
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/BCE', losses.avg(0), epoch)
            
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) BCELoss = %.4f' %
            (epoch, epoch_end_time - epoch_start_time, losses.avg(0)), logger=logger)
        
        # Validation
        if epoch % args.val_freq == 0:
            seen_metrics = validate_2d_grasp_stability(grasp_model, "Seen", val_dataloader, epoch, criterion, val_writer, args, config, logger=logger)
            metrics = validate_2d_grasp_stability(grasp_model, "Unseen", test_dataloader, epoch, criterion, test_writer, args, config, logger=logger)
            
            if metrics is not None:
                print_log('[Validation] EPOCH: %d Cur-F1 = %.3f Pre-Best-F1 = %.3f' % 
                    (epoch, metrics.F1, best_metrics.F1 if best_metrics is not None else 0), logger=logger)
                
                # Save best model
                if best_metrics is None or metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(grasp_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
        
        # Save last checkpoint
        builder.save_checkpoint(grasp_model, optimizer, epoch, metrics if 'metrics' in locals() else None, 
                               best_metrics, 'ckpt-last', args, logger=logger)
        
        # Save recent checkpoints
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(grasp_model, optimizer, epoch, metrics if 'metrics' in locals() else None,
                                   best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger=logger)
    
    # Close writers
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    if test_writer is not None:
        test_writer.close()


def validate_2d_grasp_stability(model, val_cls, val_dataloader, epoch, criterion, val_writer, args, config, logger=None):
    """
    Validation function for 2D grasp stability prediction
    """
    print_log(f"[{val_cls} VALIDATION] Start validating epoch {epoch} for Class {val_cls}", logger=logger)
    
    model.eval()
    val_losses = AverageMeter(['BCELoss'])
    
    # Accumulate predictions and ground truth for metrics calculation per object
    object_logits = dict()
    object_gts = dict()
    n_samples = len(val_dataloader)
    interval = max(1, n_samples // 10)
    pos_cnt = 0
    neg_cnt = 0
    
    with torch.no_grad():
        for idx, item in enumerate(tqdm(val_dataloader, desc="Validation Progress", unit="batch")):
            if config.grasp_model.NAME == 'GraspStability_CNNMCA':
              # Unpack 2D data: visual and tactile images
              visual_rgb = item['visual_rgb'].cuda()  # [B, 3, 224, 224]
              tactile_rgb = item['tactile_rgb_concat'].cuda() # [B, 3, 224, 224]
              grasp_result = item['grasp_result'].cuda()  # [B] 
              obj_id, grasp_id = item['sample_info']  # [B, 2] - object and grasp IDs
              # Forward pass
              logits = model(visual_rgb, tactile_rgb)  # [B, num_classes]

            elif config.grasp_model.NAME == 'GraspStability_CNN':
              # Unpack 2D data for CNN model: visual RGB and separate tactile sensors
              visual_rgb = item['visual_rgb'].cuda()  # [B, 3, 224, 224]
              tactile_rgb_left = item['tactile_rgb_left'].cuda()  # [B, 3, 224, 224]
              tactile_rgb_right = item['tactile_rgb_right'].cuda()  # [B, 3, 224, 224]
              tactile_bg_left = item['tactile_bg_left'].cuda()  # [B, 3, 224, 224]
              tactile_bg_right = item['tactile_bg_right'].cuda()  # [B, 3, 224, 224]
              grasp_result = item['grasp_result'].cuda()  # [B]
              obj_id, grasp_id = item['sample_info']  # [B, 2] - object and grasp IDs
              
              # dummy tensor for cam_bef input
              dummy_tensor = torch.zeros_like(visual_rgb)  # [B, 3, 224, 224]

              logits = model(dummy_tensor, visual_rgb, 
                           tactile_bg_left, tactile_rgb_left,
                           tactile_bg_right, tactile_rgb_right)  # [B, num_classes]
              
            else:
              raise NotImplementedError(f"Model {config.grasp_model.NAME} is not supported yet in this runner.")
            
            # Handle different output formats
            if len(logits.shape) == 1 or logits.size(1) == 1:  # Single output for binary classification
                prediction_logits = logits.squeeze() if logits.size(1) == 1 else logits  # [B]
            elif logits.size(1) == 2:  # Two class outputs, use positive class
                prediction_logits = logits[:, 1]  # [B] - positive class logits
            
            # Compute loss
            loss = criterion(prediction_logits, grasp_result)
            val_losses.update([loss.item()])
            
            # Count positive and negative samples
            batch_pos_cnt = (grasp_result == 1).sum().item()
            batch_neg_cnt = (grasp_result == 0).sum().item()
            pos_cnt += batch_pos_cnt
            neg_cnt += batch_neg_cnt
            
            # Accumulate logits and gts per object
            for i, _id in enumerate(obj_id):
                if _id not in object_logits:
                    object_logits[_id] = []
                    object_gts[_id] = []
                object_logits[_id].append(prediction_logits[i].detach().cpu())
                object_gts[_id].append(grasp_result[i].detach().cpu())
            
            if (idx + 1) % interval == 0:
              # For binary classification, use sigmoid + threshold for 0/1 output
              sigmoid_vals = torch.sigmoid(prediction_logits)
              choice_str = "[" + ", ".join([f"{(val > 0.5).int().item()}" for val in sigmoid_vals[:min(3, len(sigmoid_vals))]]) + "]"
              gt_str = "[" + ", ".join([f"{gt.item():.0f}" for gt in grasp_result[:min(3, len(grasp_result))]]) + "]"
              logit_str = "[" + ", ".join([f"{logit.item():.4f}" for logit in prediction_logits[:min(3, len(prediction_logits))]]) + "]"
              prob_str = "[" + ", ".join([f"{val.item():.4f}" for val in sigmoid_vals[:min(3, len(sigmoid_vals))]]) + "]"
              
              print_log(
                'Validation[%d/%d] BCELoss = %.4f | Logit = %s | Prob = %s | Choice = %s | GT = %s' % (
                  idx + 1, n_samples, loss.item(), logit_str, prob_str, choice_str, gt_str
                ), logger=logger
              )
    
    # Print overall pos/neg counts
    print_log(f'[{val_cls} Validation] EPOCH: {epoch} PosCnt = {pos_cnt} NegCnt = {neg_cnt}', logger=logger)
    
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
    
    # TensorBoard logging
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/BCE', val_losses.avg(0), epoch)
        for i, metric in enumerate(metric_names):
            val_writer.add_scalar(f'Metric/{metric}', overall_metrics._values[i], epoch)
    
    return GraspStabilityMetrics(config.consider_metric, overall_metrics._values) if overall_metrics._values else None
