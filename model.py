import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou
from torchvision import transforms

import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def create_faster_rcnn_model(num_classes=1, device='cpu'):
    # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone from COCO dataset
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Get the number of input features for the classification head in the RoI heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the classification head with a new one for the specified number of classes
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Move the model to the specified device (e.g., 'cpu' or 'cuda')
    model.to(device)
    
    return model


def train(model, train_loader, validation_loader, num_epochs, device='cpu', log_dir='logs'):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # writer = SummaryWriter(log_dir)
    
    # custom training monitoring
    one_percent = (train_loader.dataset.__len__() / train_loader.batch_size) // 100 + 1  # Not exact but its fine for picking occasions to validate during the training process
    # one_percent = 2 used to generate output in the notebook faster than it would be normally
    tot_leng = train_loader.dataset.__len__() / train_loader.batch_size * num_epochs
    start = time.time()
    
    # training loop. Repeat for a maximum of set number of epochs
    for epoch in range(num_epochs):
        print('Starting epoch ' + str(epoch))
        count = 0
        
        # In situation with infinite memory we would not need this for loop, but we divided our dataset into batches that are suitable for our hardware. This slows down
        # The training process but enables it on systems with limited memory and/or for incredibly large datasets
        for images, targets in train_loader:
            count += 1
            images = images.to(device)
            
            # Prepare targets variable in format expected by ResNet-50
            targets = [{k: torch.as_tensor(v, dtype=torch.float32 if k == 'boxes' else torch.int64).clone().detach() for k, v in target.items()} for target in targets]

            # reset accumulation of gradients before forward pass because the step has been made already for the last batch
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets) # generate predictions
            losses = sum(loss for loss in loss_dict.values()) # accumulate all losses in the batch
            # loss_value = losses.item()

            # Backward pass and optimization
            losses.backward() # generate gradients
            optimizer.step() # modify weigths

            # Occasionaly save model state and print statistics
            if count % one_percent == 0:  # ... == 1 was used for the notebook output to get faster output
                epoch_progress = count / train_loader.dataset.__len__()
                total_progress = (epoch + epoch_progress) / num_epochs
                t = time.time() - start
                passed = f"{str(t // 3600).zfill(2)}:{str(t // 60 % 60).zfill(2)}:{str(int(t) % 60).zfill(2)}"
                finish = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() + t/total_progress))

                print(f"Epoch {epoch}, Count {count}, Epoch progress: {epoch_progress*100:.4f}%, Total progress: {total_progress*100:.4f}%")
                print(f"Time passed: {passed}, Finnish expected: {finish}, Loss: {losses.item():.4f}, Validation(?) = {evaluate(model, validation_loader, device)}")

                # save the temoprary model state to eventualy analyse it (e. g. in a different kernel) while the training is still ongoing
                torch.save(model.state_dict(), 'temp.pth')
        
        
        print('epoch ' + str(epoch) + ' finished in ' + str(time.time() - start) + ' seconds.')
        
        # TODO: criterion to break early after an epoch is complete depending on the evaluation curve, that is, if it suggests we are starting to overfit

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
    
############ NOT USED #####################################################################
def box_matching(predicted_boxes, gt_boxes, iou_threshold=0.5):
    # Compute IoU between all predicted and ground truth boxes
    iou_matrix = np.zeros((len(predicted_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(predicted_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)

    # Convert IoU values to "cost" by subtracting from 1 (higher IoU means lower cost)
    cost_matrix = 1 - iou_matrix

    # Use linear_sum_assignment to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create a dictionary to store matched pairs
    matched_pairs = {}
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            matched_pairs[i] = j

    return matched_pairs
##########################################################################################

# TODO....
# Here we define out metric used to validate the model on the validation dataset (not seen in the training process). This is NOT DONE! This should show how the author 
# thinks this should be done technically, but mathematically only the simplest idea is provided just to get the code to pass without error.
def evaluate(model, validation_loader, device='cpu'):
    model.eval()
    with torch.no_grad():
        iou = 0
        count = 0
        for images, targets in validation_loader:
            images = images.to(device)
            predicted_boxes = model(images)[0]['boxes']  # also apply non maximum suppresion
            gt_boxes = targets[0]['boxes']
            iou += box_iou(predicted_boxes, gt_boxes).mean().mean() # This has no sense, we should only select best fits, but the develing team is running out of time
            count += 1
            print(count)
            
        score = iou/count if count > 0 else 0.0
    model.train()
    return score