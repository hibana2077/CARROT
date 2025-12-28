import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys
import os
import argparse
import yaml

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.ufgvc import UFGVCDataset
from src.carrot.backbone import TimmViTPatchBackbone
from src.carrot.regions import RegionSet
from src.carrot.graph import RegionGraphBuilder
from src.carrot.operator import DiffusionOperator
from src.carrot.readout import GraphReadout
from src.carrot.head import RidgeHead
from src.carrot.attribution import training_contribution

def parse_args():
    parser = argparse.ArgumentParser(description="CARROT Experiment Runner")
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='soybean', help='Dataset name (e.g., soybean, cotton80)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='timm model name')
    
    # CARROT Hyperparameters
    parser.add_argument('--sigma_s', type=float, default=0.5, help='Spatial kernel width')
    parser.add_argument('--sigma_f', type=float, default=1.0, help='Feature kernel width')
    parser.add_argument('--diffusion_t', type=float, default=1.0, help='Diffusion time t')
    parser.add_argument('--lambda_reg', type=float, default=1.0, help='Ridge regression regularization lambda')
    
    args = parser.parse_args()
    
    # If config file is specified, load it and override defaults, but let command line args override config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update args with config values if they are not specified on command line (this is tricky with argparse)
        # A simpler approach: set defaults from config, then re-parse or manually update
        # Here we manually update if the key exists in config and was NOT explicitly set on CLI?
        # Actually, standard pattern: Config -> Overridden by CLI.
        # But argparse sets defaults. We need to know if user provided CLI arg.
        # We can use parser.set_defaults(**config) before parsing, but we already parsed.
        
        # Let's re-parse to allow CLI to override Config
        parser.set_defaults(**config)
        args = parser.parse_args()
        
    return args

def main():
    args = parse_args()
    print(f"Configuration: {vars(args)}")

    # Configuration
    DATASET_NAME = args.dataset
    BATCH_SIZE = args.batch_size
    MODEL_NAME = args.model
    SIGMA_S = args.sigma_s
    SIGMA_F = args.sigma_f
    DIFFUSION_T = args.diffusion_t
    LAMBDA_REG = args.lambda_reg
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    print(f"Initializing {DATASET_NAME} dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = UFGVCDataset(dataset_name=DATASET_NAME, split='train', transform=transform, download=True)
        # Try 'test' split, fallback to 'val' if empty or error (though UFGVC usually has test)
        try:
            test_dataset = UFGVCDataset(dataset_name=DATASET_NAME, split='test', transform=transform, download=True)
            if len(test_dataset) == 0:
                raise ValueError("Empty test set")
        except:
            print("Test split not found or empty, using 'val' split.")
            test_dataset = UFGVCDataset(dataset_name=DATASET_NAME, split='val', transform=transform, download=True)
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 2. Initialize Modules
    print("Initializing CARROT modules...")
    backbone = TimmViTPatchBackbone(MODEL_NAME, pretrained=True, freeze=True, device=device)
    graph_builder = RegionGraphBuilder(sigma_s=SIGMA_S, sigma_f=SIGMA_F)
    operator = DiffusionOperator(t=DIFFUSION_T)
    readout = GraphReadout(method='mean')
    head = RidgeHead(lambda_reg=LAMBDA_REG)

    # 3. Extract Training Features
    print("Extracting training features (G_train)...")
    G_train_list = []
    Y_train_list = []

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Extracting Train"):
            images = images.to(device)
            
            # Backbone
            out = backbone(images)
            H, P = out.H, out.P
            
            # Region Set
            regions = RegionSet(H, P)
            
            # Graph Construction
            W, L = graph_builder.build(regions)
            
            # Diffusion
            H_prime = operator.forward(H, L)
            
            # Readout
            g = readout(H_prime)
            
            G_train_list.append(g.cpu())
            Y_train_list.append(labels)

    G_train = torch.cat(G_train_list, dim=0).to(device)
    Y_train = torch.cat(Y_train_list, dim=0).to(device)
    
    print(f"Training features shape: {G_train.shape}")
    print(f"Training labels shape: {Y_train.shape}")

    # 4. Fit Head
    print("Fitting Ridge Head...")
    head.fit(G_train, Y_train)
    print("Head fitted.")

    # 5. Evaluate
    print("Evaluating on test set...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            out = backbone(images)
            H, P = out.H, out.P
            regions = RegionSet(H, P)
            W, L = graph_builder.build(regions)
            H_prime = operator.forward(H, L)
            g = readout(H_prime)
            
            logits = head.predict(g)
            predictions = torch.argmax(logits, dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # 6. Attribution Demo
    print("\n=== Attribution Demo ===")
    print("Analyzing the first image from the test set...")
    
    test_image, test_label = test_dataset[0]
    test_image = test_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = backbone(test_image)
        H, P = out.H, out.P
        regions = RegionSet(H, P)
        W, L = graph_builder.build(regions)
        H_prime = operator.forward(H, L)
        g_test = readout(H_prime) # (1, D)
        
        # Calculate influence of training samples
        # Note: Y_train is passed but not used in the current influence calculation function
        # which returns the raw influence weights.
        influence = training_contribution(g_test, G_train, Y_train, lambda_reg=LAMBDA_REG)
        
        # Find most influential training samples
        top_k = 5
        # Sort by absolute influence
        top_indices = torch.topk(torch.abs(influence), top_k).indices[0]
        
        print(f"Test Image Label: {test_label} ({test_dataset.get_class_name(0) if hasattr(test_dataset, 'get_class_name') else ''})")
        print(f"Top {top_k} influential training samples:")
        
        for idx in top_indices:
            idx_val = idx.item()
            inf_val = influence[0, idx_val].item()
            train_label = Y_train[idx_val].item() if Y_train.dim() == 1 else torch.argmax(Y_train[idx_val]).item()
            
            # Try to get class name if possible
            class_name = ""
            if hasattr(train_dataset, 'get_class_name'):
                class_name = f"({train_dataset.get_class_name(idx_val)})"
                
            print(f"  Train Sample #{idx_val}: Influence = {inf_val:.6f}, Label = {train_label} {class_name}")

if __name__ == "__main__":
    main()
