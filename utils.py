import torch
from pathlib import Path
import matplotlib.pyplot as plt

def find_lr(model, train_loader, criterion, optimizer, 
            init_lr=1e-8, final_lr=1, num_steps=25, plot=False):
    model.train()

    # Set up learning rate range
    lr_step = (final_lr / init_lr) ** (1 / num_steps)
    lr_values = [init_lr * (lr_step ** i) for i in range(num_steps)]
    losses = []

    for k, lr in enumerate(lr_values):
        print(f"\n\nStep {k} of {len(lr_values)}\n\n")
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Train for one mini-batch
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Record the loss
        losses.append(loss.item())

    if plot==True:
      plt.plot(lr_values, losses)
      plt.xscale('log')
      plt.xlabel('Learning Rate')
      plt.ylabel('Loss')
      plt.grid(True)

    return lr_values, losses

def plot_history_curves(train_history, test_history):
  epochs = range(len(train_history))
  if len(epochs) == 0:
    return None

  for metric in train_history[0].keys():
    plt.figure()
    train_metric = [train_history[epoch][metric] for epoch in epochs]
    test_metric = [test_history[epoch][metric] for epoch in epochs]
    plt.plot(epochs, train_metric, marker='o', ms=3, markeredgecolor='k', label=f'Train {metric}')
    plt.plot(epochs, test_metric, marker='o', ms=3, markeredgecolor='k', label=f'Test {metric}')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel(metric)

def visualize_preds(model, test_batch):
  X_test, y_test = test_batch
  n_samples = len(X_test)

  plt.figure(figsize=(16, 32))
  model.eval()
  with torch.inference_mode():
    preds = model(X_test)
    preds_labels = torch.argmax(F.softmax(preds, dim=-1), dim=-1)

    for k in range(batch_size):
      plt.subplot(n_samples//4,4,k+1)
      plt.imshow(X_test[k].permute(1,2,0))
      title_str = f"Pred={train_data.classes_[preds_labels[k].item()]}, True={train_data.classes_[y_test[k].item()]}"
      if preds_labels[k] == y_test[k]:
        plt.title(title_str, color='green')
      else:
        plt.title(title_str, color='red')

def save_model(model, target_dir, model_name):
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  model_save_path = target_dir_path / model_name
  torch.save(obj=model.state_dict(), f=model_save_path)
