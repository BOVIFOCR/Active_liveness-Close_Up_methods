import torch

def run_one_epoch(model,dataset_loader,criterion,optimizer,splitInputs=False,device=None):
    
    overall_loss = 0.0
    correct = 0
    total = 0
    steps = 0
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for inputs, labels in dataset_loader:
        if optimizer is not None:
            # zero the parameter gradients
            optimizer.zero_grad()

        labels = labels.unsqueeze(1)
        labels = labels.float()
        if splitInputs and device is not None:
            inputs = (inputs[0].to(device),inputs[1].to(device))
            labels = labels.to(device)
        elif device is not None:
            inputs = inputs.to(device)
            labels = labels.to(device)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        overall_loss += loss.item() * labels.size(0)
        
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        # compute statistics
        predicted = torch.sigmoid(outputs)
        predicted = predicted.round()
        total += labels.size(0)
        steps += 1
        correct += (predicted == labels).sum().item()
        tp += ((predicted == 1) & (labels == 1)).sum()
        tn += ((predicted == 0) & (labels == 0)).sum()
        fp += ((predicted == 1) & (labels == 0)).sum()
        fn += ((predicted == 0) & (labels == 1)).sum()

    # Compute Train loss, HTER, Precision, Recall, and F1-score
    overall_loss = overall_loss/steps
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    far = fp/(fp+tn)
    frr = fn/(fn+tp)

    # return loss, accuracy, HTER, F1-score
    return overall_loss,(100*correct/total),(100*(far + frr)/2),f1

