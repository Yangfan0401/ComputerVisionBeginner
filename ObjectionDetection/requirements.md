# Faster-RCNN

## 1. Backbone

FPN多尺度特征提取

**Input**: (BatchSize, channels, H, W) $\rightarrow$ (16, 3, 224, 224)
**Ouput**: Json each (BatchSize, channels, H, W) 
* p3: (16, 64, 28, 28) stride 8
* p4: (16, 64, 14, 14) stride 16
* p5: (16, 64, 7, 7) stride 32

## 2. One Stage(RPN Region Proposal Network)

### 2.1 RPNPredictionNetwork
CNN生成各层特征的 目标与回归框偏置 预测
**Input**: fpn_feats
**Ouput**: 
* pred_object_logits (BatchSize, H\*W\*num_anchors) 
* pred_boxreg_deltas (BatchSize, H\*W\*num_anchors, 4) 

**num_anchors** 为所有尺度累计的anchors数量

### 2.2 AnchorsGeneration
#### 1. location_coords

**Input**: 

* fpn_shape
* fpn_strides

**Output**: fpn_location_coords(Json each location_coords)
fpn各层 逐像素网格中心点 按stride 投影到原始图像坐标

* location_coords: (H\*W, 2)


#### 2. generate_anchors

**Input**:

* fpn_location_coords
* fpn_stride
* stride_scale 设定的anchors尺寸(scale)  **调节anchors大小**
* aspect_ratio 设定的anchors长宽比 **调节anchors形状**

**Output**: fpn_anchors(Json each )

以location_coords为中心点 绘制anchors 

预设每张图的anchors位置大小是一致的

```python
area = (stride_scale * level_stride)**2 # anchor面积
width = math.sqrt(area / aspect_ratio) # anchor宽
height = area / new_width # anchor长
```

* anchor_boxes: (A, H\*W, 4)

`A` = per_fpn_num_anchors= (len(stride_scale)*len(aspect_ratio))


### 2.3 PredictProposals

**Input**:
* fpn_anchors
* pred_obj_logits 
* pred_boxreg_deltas **预测的是对anchors的偏移**
* image_size 原始图像尺寸
#### 1. ProposalBoxes

按论文公式计算ProposalBox位置

每张图的anchors是一致的 但是ProposalBoxes是不一样的 因此Proposal

#### 2. 优化

1. proposals裁剪 坐标上下限
2. 每次训练初始的ProposalBoxes往往低质量的 可以按obj_logits去 筛选
* topk 
* nms
* topk

### 2.4 Matched GT Boxes

* AnchorBoxes match to GT Boxes

计算Anchor Boxes与GT Boxes计算按iou_thresh

Anchors having high IoU with any GT box are assigned "foreground" and matched with that box or vice-versa.

### 2.5 Loss

#### 1. Sample

按fg与bg采样 取 anchor_boxes、matched_gt_boxes、pred_obj_logits、pred_boxreg_deltas

#### 2. loss

1. loss_boxes

anchor_boxes与matched_gt_boxes计算gt_deltas
gt_deltas与pred_boxreg_deltas计算l1_loss

2. loss_obj

matched_gt_boxes取前景布尔结果(前景True 背景False) 与 pred_obj_logits计算二元交叉熵逻辑损失

## 3. Two Stage

### 3.1 Mix gt with proposals

Mix GT boxes with proposals. This is necessary to stabilize training since RPN proposals may be bad during first few iterations.
往Proposals中掺一点GT Boxes提高Proposals质量

### 3.2 Cls_pred with ROI Pooling

把per_fpn_proposals尺度还原回原图像尺度 得到proposals_feat
再用CNN做一个计算得到分类预测值
**Output**:pred_cls_logits (BatchSize * total_proposals, num_classes)

### 3.3 Matched GT Boxes with proposals

### 3.4 Loss

采样 算分类损失