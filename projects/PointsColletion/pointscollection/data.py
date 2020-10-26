import torch
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from detectron2.structures import Boxes,Instances

import matplotlib.pyplot as plt

CONTOUR_SIZE=81
class Targets:
    def __init__(self,cfg,pc_stride,cls_stride):
        
        self.pc_stride=pc_stride
        self.cls_stride=cls_stride
        self.num_class=cfg.MODEL.POINTS_COLLECTION.NUM_CLASSES
        self.sigma=cfg.MODEL.POINTS_COLLECTION.SIGMA
        self.contour=cfg.MODEL.POINTS_COLLECTION.CONTOUR
        self.mask_on=cfg.MODEL.POINTS_COLLECTION.MASK_ON
        assert self.pc_stride==self.cls_stride,"cls stride should be equal to pc's!"

    def get_target_single(self,classes,bitmask,output_size):
        bitmask=bitmask.tensor.cpu().numpy()
        classes=classes.cpu().numpy()
        N=classes.shape[0]
        # print(classes)

        H_out=output_size[2]
        W_out=output_size[3]
        cls_target=np.zeros((self.num_class,H_out,W_out),dtype=np.float32)   
        
        offsets=[]
        belongs=[]
        for i in range(N):
            this_bitmask=bitmask[i]
            center,count=self.fill_expand_points_center(np.uint8(this_bitmask))

            # plt.imshow(this_bitmask)
            # plt.scatter(count[:,0],count[:,1])
            # plt.show()
            # print(bitmask.shape)
            # print(count,self.pc_stride)

            digit=classes[i]

            resize_center=center/self.cls_stride
            resize_count=count/self.pc_stride
            cls_target[digit],belong=self.generate_onehot(cls_target[digit],resize_center)
            if belong.shape[0]>0:
                resize_count=np.expand_dims(resize_count,0)
                # print(resize_count.shape,belong.shape)
                resize_count_xy2yx=np.zeros_like(resize_count)
                resize_count_xy2yx[0,:,0]=resize_count[0,:,1]
                resize_count_xy2yx[0,:,1]=resize_count[0,:,0]
                resize_count_xy2yx=np.tile(resize_count_xy2yx,(belong.shape[0],1,1))
                expand_belong=np.expand_dims(belong,1)
                offset=resize_count_xy2yx-expand_belong
                # print(offset,belong,list(resize_count_xy2yx[0]))
                offsets.append(offset)
                belongs.append(belong)
        
        pc_target_belongs=np.concatenate(belongs) if len(belongs)>0 else np.zeros((0,2),dtype=np.int64)
        # pc_target_belongs=np.int64(pc_target_belongs)
        pc_target_offsets=np.concatenate(offsets) if len(offsets)>0 else np.zeros((0,self.contour,2),dtype=np.float32)

        if self.mask_on:
            return cls_target,pc_target_belongs,pc_target_offsets,bitmask

        return cls_target,pc_target_belongs,pc_target_offsets



    def generate_heatmap(self,heatmap,center):
        tmp_size = self.sigma
        mu_x = int(center[0] + 0.5)
        mu_y = int(center[1] + 0.5)
        h, w = heatmap.shape[0], heatmap.shape[1]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
            return heatmap,np.zeros((0,2))
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], w)
        img_y = max(0, ul[1]), min(br[1], h)
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]])

        belongs=np.argwhere(heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]]==g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        if belongs.shape[0]>0:
            belongs[:,0]+=img_y[0]
            belongs[:,1]+=img_x[0]

        return heatmap,belongs

    def generate_onehot(self,heatmap,center):  
        mu_x = int(center[0] + 0.5)
        mu_y = int(center[1] + 0.5)
        h, w = heatmap.shape[0], heatmap.shape[1]
        img_x = min(max(0, mu_x),w-1)
        img_y = min(max(0, mu_y), h-1)
        heatmap[img_y, img_x] =1.0
        belongs=np.zeros((1,2),dtype=np.int64)
        belongs[0,0]=img_y
        belongs[0,1]=img_x
        return heatmap,belongs



    def get_centerpoint(self, lis):
        area = 0.0
        x, y = 0.0, 0.0
        a = len(lis)
        for i in range(a):
            lat = lis[i][0]
            lng = lis[i][1]
            if i == 0:
                lat1 = lis[-1][0]
                lng1 = lis[-1][1]
            else:
                lat1 = lis[i - 1][0]
                lng1 = lis[i - 1][1]
            fg = (lat * lng1 - lng * lat1) / 2.0
            area += fg
            x += fg * (lat + lat1) / 3.0
            y += fg * (lng + lng1) / 3.0
        x = x / area
        y = y / area

        return [int(x), int(y)]

    def fill_expand_polygon_center(self,bitmask):
        contour, _ = cv2.findContours(bitmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) #only save the biggest one
        '''debug IndexError: list index out of range'''
        count = contour[0][:, 0, :]
        try:
            center = self.get_centerpoint(count)
        except:
            x,y = count.mean(axis=0)
            center=[int(x), int(y)]

        #make count has the same length
        new_count=np.zeros((self.contour,2),dtype=np.float32)
        length=count.shape[0]
        if length<2:
            # print(contour)
            # plt.imshow(bitmask)
            # plt.scatter(count[:,0],count[:,1])
            # plt.show()
            new_count[:,0]=count[0,0]
            new_count[:,1]=count[0,1]
            return np.array(center),new_count
        interval=1.0*(length-1)/(self.contour-1)    
        z=[x*interval for x in range(self.contour)]
        z_org=range(length)
        # print(count.shape)
        interx=interp1d(z_org,count[:,0],bounds_error=False,fill_value="extrapolate")
        newx=interx(z)
        intery=interp1d(z_org,count[:,1],bounds_error=False,fill_value="extrapolate")
        newy=intery(z)
        new_count[:,0]=newx
        new_count[:,1]=newy
      
        return np.array(center),new_count

    def fill_expand_points_center(self,bitmask):
        contour, _ = cv2.findContours(bitmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) #only save the biggest one
        '''debug IndexError: list index out of range'''
        count = contour[0][:, 0, :]
        try:
            center = self.get_centerpoint(count)
        except:
            x,y = count.mean(axis=0)
            center=[int(x), int(y)]

        ISize=self.contour-CONTOUR_SIZE
        new_count=np.zeros((CONTOUR_SIZE,2),dtype=np.float32)
        new_inner_points=np.zeros((ISize,2),dtype=np.float32)
        length=count.shape[0]
        if length<2:
            # print(contour)
            # plt.imshow(bitmask)
            # plt.scatter(count[:,0],count[:,1])
            # plt.show()
            new_count[:,0]=count[0,0]
            new_count[:,1]=count[0,1]
            new_inner_points[:,0]=count[0,0]
            new_inner_points[:,1]=count[0,1]
            return np.array(center),np.concatenate([new_count,new_inner_points],axis=0)

        interval=1.0*(length-1)/(CONTOUR_SIZE-1)    
        z=[x*interval for x in range(CONTOUR_SIZE)]
        z_org=range(length)
        # print(count.shape)
        interx=interp1d(z_org,count[:,0],bounds_error=False,fill_value="extrapolate")
        newx=interx(z)
        intery=interp1d(z_org,count[:,1],bounds_error=False,fill_value="extrapolate")
        newy=intery(z)
        new_count[:,0]=newx
        new_count[:,1]=newy

        new_mask=bitmask.copy()
        new_mask[:]=0
        cv2.fillPoly(new_mask,[count,],(255,))
        inner_points=np.argwhere(new_mask>100)
        inner_points_yx2xy=inner_points[:,(1,0)]
        inner_points_size=inner_points_yx2xy.shape[0]
        # print(inner_points_size,ISize)
        if inner_points_size<ISize: 
            extra_size=ISize-inner_points_size      
            index=np.random.choice(inner_points_size,extra_size, replace=True) 

            extra=inner_points_yx2xy[index,:]+(np.random.rand(extra_size,2)-0.5)
            new_inner_points[:inner_points_size,:]=inner_points_yx2xy
            new_inner_points[inner_points_size:,:]=extra
        else:
            index=np.random.choice(inner_points_size, ISize, replace=False)
            new_inner_points[:,:]=inner_points_yx2xy[index,:]

        return np.array(center),np.concatenate([new_count,new_inner_points],axis=0)



        



def _postprocess(results, output_height, output_width,):
    """
    Post-process the output boxes for TensorMask.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will postprocess the raw outputs of TensorMask
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place. Note that it does not contain the field
            `pred_masks`, which is provided by another input `result_masks`.
        result_mask_info (list[Tensor], Boxes): a pair of two items for mask related results.
                The first item is a list of #detection tensors, each is the predicted masks.
                The second item is the anchors corresponding to the predicted masks.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the postprocessed output from the model, based on the output resolution
    """
    # old_image_size=results.image_size
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    output_boxes = results.pred_boxes
    output_boxes.tensor[:, 0::2] *= scale_x
    output_boxes.tensor[:, 1::2] *= scale_y
    output_boxes.clip(results.image_size)

    pred_points=results.pred_points
    pred_points[:,:, 0] *= scale_x
    pred_points[:,:, 1] *= scale_y
    h,w=results.image_size
    pred_points[:,:, 0].clamp_(min=0, max=w)
    pred_points[:,:, 1].clamp_(min=0, max=h)

    if results.has('pred_masks'):
        old_pred_masks=results.pred_masks
        pred_masks=torch.nn.functional.interpolate(old_pred_masks.unsqueeze(1).float(),(output_height,output_width),mode='nearest').bool().squeeze(1)
        # print(old_pred_masks.size(),pred_masks.size())
    else:
        pred_masks=points_to_masks(pred_points,results.image_size)
    

    results.pred_masks = pred_masks
    # results.pred_points = pred_points
   
    return results


def points_to_masks(pred_points,image_size):

    N=pred_points.size(0)
    if N <1:
        return torch.zeros(0,800,800,dtype=torch.bool)
    masks=[]
    for i in range(N):
        points=pred_points[i].cpu().numpy()
        points=np.int32(points)
        contour=cv2.convexHull(points)
        img=np.zeros(image_size,np.uint8)
        mask=cv2.fillPoly(img, [contour[:,0,:]], (255))
        mask=torch.from_numpy(mask).to(device=pred_points.device)
        mask=(mask>0).to(dtype=torch.bool)
        masks.append(mask)

    return torch.stack(masks,dim=0)
    