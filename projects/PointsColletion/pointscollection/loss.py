import torch
import torch.nn as nn
import torch.nn.functional as F 
from pointscollection.layers.emd import emd_function


def chamfer_loss(pred_points,gt_points):
    p2pdistance=torch.sum((pred_points-gt_points)**2,dim=2)
    dist1,_=torch.min(p2pdistance,dim=1)
    dist2,_=torch.min(p2pdistance,dim=2)

    dist1=dist1.mean(-1)
    dist2=dist2.mean(-1)
    dist=(dist1+dist2)/2.0

    return torch.mean(dist)
    
def normlize_chamfer_loss(pred_points,gt_points,max_side=32):
    eps=10e-5
    with torch.no_grad():
        pred_points_copy=pred_points.detach()
        gt_points_copy=gt_points.detach()

        p_mean=torch.mean(pred_points_copy,dim=1,keepdim=True)
        g_mean=torch.mean(gt_points_copy,dim=3,keepdim=True)

        p_align=pred_points_copy-p_mean
        g_align=gt_points_copy-g_mean

        ## this is max side alignment
        p_norm=torch.abs(p_align)
        p_norm,_=torch.max(p_norm,dim=1,keepdim=True)

        g_norm=torch.abs(g_align)
        g_norm,_=torch.max(g_norm,dim=3,keepdim=True)

        p_norm=torch.clamp(p_norm, min=eps,max=max_side)
        g_norm=torch.clamp(g_norm,min=eps,max=max_side)

        p_align_new=p_align*g_norm/p_norm

        distance=torch.sum((p_align_new-g_align)**2,dim=2,keepdim=True)
        _,min_index_gt=torch.min(distance,dim=3,keepdim=True)
        _,min_index_pt=torch.min(distance,dim=1,keepdim=True)
        rep_min_index_gt=min_index_gt.repeat(1,1,2,1)
        rep_min_index_pt=min_index_pt.repeat(1,1,2,1)
            
    tran_gt_points=torch.transpose(gt_points,1,3)
    gather_gt_points=torch.gather(tran_gt_points,1,rep_min_index_gt)

    tran_pt_points=torch.transpose(pred_points,1,3)
    gather_pt_points=torch.gather(tran_pt_points,3,rep_min_index_pt)

    # dist1=torch.sum((pred_points-gather_gt_points)**2,dim=2).squeeze()
    # dist1=F._smooth_l1_loss(pred_points,gather_gt_points).squeeze(3)
    # dist2=torch.sum((gather_pt_points-gt_points)**2,dim=2).squeeze()
    # dist2=F._smooth_l1_loss(gather_pt_points,gt_points).squeeze(1)
    # dist1=torch.sum((pred_points-gather_gt_points)**2,dim=2).squeeze()
    # dist2=torch.sum((gather_pt_points-gt_points)**2,dim=2).squeeze()
    dist1=torch.abs(pred_points-gather_gt_points).squeeze()
    dist2=torch.abs(gather_pt_points-gt_points).squeeze()

    dist1=dist1.mean(-2)
    dist2=dist2.mean(-1)
    dist=(dist1+dist2)/2.0

    return torch.mean(dist)    
        

def outlier_loss(pred_points,gt_points,contour_size=81):

    npoints=gt_points.size(3)
    inner_size=npoints-contour_size
    contour,inner=torch.split(gt_points,[contour_size,inner_size],dim=3)

    dist1=torch.sum((pred_points-contour)**2,dim=2)
    mindist1,mindist1_index=torch.min(dist1,dim=2)

    dist2=torch.sum((pred_points-inner)**2,dim=2)
    mindist2,mindist2_index=torch.min(dist2,dim=2)


    penalty=torch.where(dist1<dist2,dist1,0)

    return torch.mean(penalty)


def normlize_chamfer_loss_with_outlier_penalty(pred_points,gt_points,contour_size=81,max_side=32):
    eps=10e-5
    with torch.no_grad():
        pred_points_copy=pred_points.detach()
        gt_points_copy=gt_points.detach()

        p_mean=torch.mean(pred_points_copy,dim=1,keepdim=True)
        g_mean=torch.mean(gt_points_copy,dim=3,keepdim=True)

        p_align=pred_points_copy-p_mean
        g_align=gt_points_copy-g_mean

        ## this is max side alignment
        p_norm=torch.abs(p_align)
        p_norm,_=torch.max(p_norm,dim=1,keepdim=True)

        g_norm=torch.abs(g_align)
        g_norm,_=torch.max(g_norm,dim=3,keepdim=True)

        p_norm=torch.clamp(p_norm, min=eps,max=max_side)
        g_norm=torch.clamp(g_norm,min=eps,max=max_side)

        p_align_new=p_align*g_norm/p_norm

        distance=torch.sum((p_align_new-g_align)**2,dim=2,keepdim=True)
        _,min_index_gt=torch.min(distance,dim=3,keepdim=True)
        _,min_index_pt=torch.min(distance,dim=1,keepdim=True)
        rep_min_index_gt=min_index_gt.repeat(1,1,2,1)
        rep_min_index_pt=min_index_pt.repeat(1,1,2,1)

        outlier_index=(min_index_gt<contour_size).squeeze()
            
    tran_gt_points=torch.transpose(gt_points,1,3)
    gather_gt_points=torch.gather(tran_gt_points,1,rep_min_index_gt)

    tran_pt_points=torch.transpose(pred_points,1,3)
    gather_pt_points=torch.gather(tran_pt_points,3,rep_min_index_pt)

    dist1=torch.sum((pred_points-gather_gt_points)**2,dim=2).squeeze()

    outlier_penalty=dist1[outlier_index]
    dist2=torch.sum((gather_pt_points-gt_points)**2,dim=2).squeeze()

    dist1=dist1.mean(-1)
    dist2=dist2.mean(-1)
    dist=(dist1+dist2)/2.0

    return torch.mean(dist),torch.mean(outlier_penalty)   


def emd_loss(pred_points,gt_points,eps=0.005,iters=50):

    pred_points=pred_points.squeeze(3)
    gt_points=gt_points.squeeze(1)
    gt_points=gt_points.transpose(1,2)

    dist,_=emd_function(pred_points,gt_points,eps,iters)

    return torch.mean(dist)


def emd_l1_loss(pred_points,gt_points,eps=0.005,iters=50):

    pred_points=pred_points.squeeze(3)
    gt_points=gt_points.squeeze(1)
    gt_points=gt_points.transpose(1,2)

    _,assignment=emd_function(pred_points,gt_points,eps,iters)

    assignment=assignment.unsqueeze(2)
    assignment=assignment.repeat(1,1,2)
    gt_points=torch.gather(gt_points,1,assignment)

    dist=torch.abs(x1-x2)
    dist=dist.mean(-1)

    return torch.mean(dist)

