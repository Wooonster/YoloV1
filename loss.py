import torch
import torch.nn as nn
from utils import intersection_over_union as IOU

class Loss(nn.Module):
  def __init__(self, S=7, B=2, C=20):
    super().__init__()
    self.mse = nn.MSELoss(reduction="sum")
    self.S, self.B, self.C = S, B, C
    self.lambda_noobj, self.lambda_coord = .5, 5

  def forward(self, predictions, target):
    predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

    # predictions[..., ] == predictions[:, : , :, ]
    # predictions[..., 21:25]: x, y, w, h for 1st bounding box, 25 -> confidence
    iou_b1 = IOU(predictions[..., 21:25], target[..., 21:25])
    # predictions[..., 26:30]: x, y, w, h for 2nd bounding box
    iou_b2 = IOU(predictions[..., 26:30], target[..., 21:25])
    ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
    # iou_maxes -> max value, best_box -> argmax
    iou_maxes, best_box = torch.max(ious, dim=0)
    # Iobj_i -> 0 or 1, is there an object in cell i
    exists_box = target[..., 20].unsqueeze(3)  

    '''
    box coordinates loss
    '''
    box_predictions = exists_box * (
      (
        best_box * predictions[..., 26:30] 
        + (1 - best_box) * predictions[..., 21:25]
      )
    )
    box_target = exists_box * target[..., 21:25]

    # box_predictions[..., 2:4] -> w, h
    # torch.abs() to avoid sign error
    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
      torch.abs(box_predictions[..., 2:4]+ 1e-6)  # avoid infinite
    )
    box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

    box_loss = self.mse(
      # flatten all dimensions before, (N, S, S, 4) -> (N*S*S, 4)
      torch.flatten(box_predictions, end_dim=-2),
      torch.flatten(box_target, end_dim=-2),
    )

    '''
    object loss
    '''
    pred_box = (
      best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
    )

    object_loss = self.mse(
      torch.flatten(exists_box * pred_box),
      torch.flatten(exists_box * target[..., 20:21])
    )


    '''
    no object loss
    '''
    no_obj_loss = self.mse(
      torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
      torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
    )

    no_obj_loss += self.mse(
      torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
      torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
    )

    '''
    class loss
    '''
    class_loss = self.mse(
      torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
      torch.flatten(exists_box * target[..., :20], end_dim=-2)
    )

    loss = (
      self.lambda_coord * box_loss
      + object_loss
      + self.lambda_noobj * no_obj_loss
      + class_loss
    )

    return loss
