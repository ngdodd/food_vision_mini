import torch
from torchvision import transforms
from torchmetrics import Accuracy
import data_setup, engine, model_builder, utils

def train(model=None, config=None):
  torch.manual_seed(42)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  if config == None:
    config = {}
    config['lr'] = 0.005
    config['batch_size'] = 32
    config['img_size'] = 128
    config['n_epochs'] = 40
    config['model_name'] = 'vgg_med_normalized'
    config['device'] = device

  # Prepare datasets and data loaders
  train_transforms = transforms.Compose([
      transforms.Resize((config['img_size'], config['img_size'])),
      transforms.TrivialAugmentWide(num_magnitude_bins=31),
      transforms.ToTensor()
  ])

  test_transforms = transforms.Compose([
      transforms.Resize((config['img_size'], config['img_size'])),
      transforms.ToTensor()
  ])

  train_dl, test_dl, class_names = data_setup.run_all(batch_size=config['batch_size'],
                                                      train_transforms=train_transforms,
                                                      test_transforms=test_transforms)

  # Build model
  if model == None:
    model = model_builder.MediumVGGNormalized(n_classes=len(class_names),
                                              image_height=config['img_size'],
                                              image_width=config['img_size']).to(config['device'])

  # Train model
  metrics = {'accuracy': Accuracy(task='multiclass', num_classes=3).to(config['device'])}
  train_history, test_history = engine.train_model(model=model, 
                                                   train_dl=train_dl, 
                                                   test_dl=test_dl, 
                                                   metrics=metrics, 
                                                   device=config['device'], 
                                                   n_epochs=config['n_epochs'], 
                                                   lr=config['lr']
  )

  utils.plot_history_curves(train_history, test_history)
  utils.save_model(model, 
                  target_dir='/content/going_modular/models', 
                  model_name=config['model_name'])
