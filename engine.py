import copy
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

def init_results(metrics):
  results_map = {k: 0.0 for k in metrics.keys()}
  results_map['loss'] = 0.0
  return results_map

def preds_transform(y_preds):
  return torch.argmax(F.softmax(y_preds, dim=-1), dim=-1)

def train_step(model, X, y, loss_fn, optimizer, results, preds_transform=None, metrics={}):
  model.train()
  y_preds = model(X)
  loss = loss_fn(y_preds, y)

  for metric, metric_fn in metrics.items():
    if preds_transform != None:
      y_preds = preds_transform(y_preds)
    results[metric] += metric_fn(y_preds, y).item()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  results['loss'] += loss.item()

def test_step(model, X, y, loss_fn, results, preds_transform=None, metrics={}):
  model.eval()
  with torch.inference_mode():
    y_preds = model(X)
    loss = loss_fn(y_preds, y)

    for metric, metric_fn in metrics.items():
      if preds_transform != None:
        y_preds = preds_transform(y_preds)
      results[metric] += metric_fn(y_preds, y).item()

    results['loss'] += loss.item()

def train_model(model, train_dl, test_dl, metrics, device, n_epochs=5, lr=0.001):
  loss_fn = nn.CrossEntropyLoss()
  #optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9)
  optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
  #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
  lr_scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl), epochs=n_epochs)
  train_history, test_history = [], []

  for epoch in range(n_epochs):
    train_results, test_results = init_results(metrics), init_results(metrics)
    for X_batch, y_batch in train_dl:
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      train_step(model, X_batch, y_batch, loss_fn, optimizer, train_results, preds_transform, metrics)
      lr_scheduler.step()

    for X_batch, y_batch in test_dl:
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      test_step(model, X_batch, y_batch, loss_fn, test_results, preds_transform, metrics)

    # Aggregate, report, etc
    train_results = {k:v/len(train_dl) for k,v in train_results.items()}
    test_results = {k:v/len(test_dl) for k,v in test_results.items()}
    train_history.append(copy.deepcopy(train_results))
    test_history.append(copy.deepcopy(test_results))
    print(f"Train Results: {train_results}")
    print(f"Test Results: {test_results}")

    #lr_scheduler.step()
  return train_history, test_history
