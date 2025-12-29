import torch
import torch.nn as nn
import torch.optim as optim
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
    parser.add_argument('--fine_tune_backbone', action='store_true', help='Fine-tune the backbone before CARROT extraction')
    parser.add_argument('--ft_epochs', type=int, default=10, help='Number of epochs for backbone fine-tuning')
    parser.add_argument('--ft_lr', type=float, default=1e-4, help='Learning rate for backbone fine-tuning')
    
    # CARROT Hyperparameters
    parser.add_argument('--sigma_s', type=float, default=0.5, help='Spatial kernel width')
    parser.add_argument('--sigma_f', type=float, default=1.0, help='Feature kernel width')
    parser.add_argument('--diffusion_t', type=float, default=1.0, help='Diffusion time t')
    parser.add_argument('--lambda_reg', type=float, default=1.0, help='Ridge regression regularization lambda')
    
    # Attribution
    parser.add_argument('--analyze_attribution', action='store_true', default=True, help='Run attribution analysis')
    parser.add_argument('--attribution_limit', type=int, default=-1, help='Number of test samples to analyze (-1 for all)')
    parser.add_argument('--top_k', type=int, default=5, help='Top K influential samples to retrieve')
    
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
    # If fine-tuning, we start unfrozen. Otherwise frozen.
    init_freeze = not args.fine_tune_backbone
    backbone = TimmViTPatchBackbone(MODEL_NAME, pretrained=True, freeze=init_freeze, device=device)
    graph_builder = RegionGraphBuilder(sigma_s=SIGMA_S, sigma_f=SIGMA_F)
    operator = DiffusionOperator(t=DIFFUSION_T)
    readout = GraphReadout(method='mean')
    head = RidgeHead(lambda_reg=LAMBDA_REG)

    # 2.5 Fine-tune Backbone (Optional)
    if args.fine_tune_backbone:
        print(f"Fine-tuning backbone for {args.ft_epochs} epochs...")
        
        # Determine embedding dim
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out = backbone(dummy_input)
        embed_dim = out.H.shape[-1]
        
        # Determine num classes
        try:
            num_classes = len(train_dataset.classes)
        except:
            all_labels = [y for _, y in train_dataset]
            num_classes = len(set(all_labels))
            
        print(f"Fine-tuning with {num_classes} classes.")
        ft_head = nn.Linear(embed_dim, num_classes).to(device)
        
        # Optimizer for backbone + head
        optimizer = optim.AdamW(list(backbone.model.parameters()) + list(ft_head.parameters()), lr=args.ft_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training Loop
        backbone.model.train()
        ft_head.train()
        
        # We need a shuffled loader for training
        ft_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
        
        for epoch in range(args.ft_epochs):
            total_loss = 0
            correct = 0
            total = 0
            pbar = tqdm(ft_loader, desc=f"Fine-tuning Epoch {epoch+1}/{args.ft_epochs}")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward
                out = backbone(images) # (B, N, D)
                # Mean pool for classification
                features = out.H.mean(dim=1) # (B, D)
                logits = ft_head(features)
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({'loss': total_loss/total, 'acc': correct/total})
        
        print("Fine-tuning complete. Freezing backbone.")
        # Freeze backbone for CARROT
        backbone.model.eval()
        for p in backbone.model.parameters():
            p.requires_grad = False

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
    
    # Store test features and labels for attribution
    G_test_list = []
    Y_test_list = []
    
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
            
            if args.analyze_attribution:
                G_test_list.append(g.cpu())
                Y_test_list.append(labels.cpu())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # 6. Attribution Analysis
    if args.analyze_attribution:
        print("\n=== Attribution Analysis ===")
        
        G_test = torch.cat(G_test_list, dim=0).to(device)
        Y_test = torch.cat(Y_test_list, dim=0).to(device)
        
        num_test = G_test.size(0)
        limit = args.attribution_limit if args.attribution_limit > 0 else num_test
        limit = min(limit, num_test)
        
        print(f"Analyzing attribution for {limit} test samples...")
        
        # Metrics
        top_k_consistency = 0.0
        
        # Process in batches to avoid OOM if N_test * N_train is too large
        # But for simplicity, we iterate one by one or small chunks if needed.
        # Given UFGVC sizes, we can probably do a loop.
        
        results = []
        
        for i in tqdm(range(limit), desc="Attribution"):
            g_i = G_test[i:i+1] # (1, D)
            y_i = Y_test[i].item()
            
            # Calculate influence: (1, N_train)
            influence = training_contribution(g_i, G_train, Y_train, lambda_reg=LAMBDA_REG)
            
            # Get Top-K
            # Sort by absolute influence
            top_indices = torch.topk(torch.abs(influence), args.top_k).indices[0] # (K,)
            
            # Check consistency
            # Note: Y_train might be one-hot or indices. 
            # In head.fit, we converted Y_train to one-hot if it was 1D.
            # So Y_train here is likely one-hot (N_train, C).
            
            current_consistency = 0
            retrieved_samples = []
            
            for idx in top_indices:
                idx_val = idx.item()
                inf_val = influence[0, idx_val].item()
                
                # Get label of training sample
                if Y_train.dim() > 1:
                    train_label = torch.argmax(Y_train[idx_val]).item()
                else:
                    train_label = Y_train[idx_val].item()
                
                if train_label == y_i:
                    current_consistency += 1
                
                retrieved_samples.append({
                    'train_idx': idx_val,
                    'influence': inf_val,
                    'label': train_label
                })
            
            top_k_consistency += (current_consistency / args.top_k)
            
            results.append({
                'test_idx': i,
                'test_label': y_i,
                'top_k_samples': retrieved_samples
            })
            
        avg_consistency = top_k_consistency / limit
        print(f"\nAttribution Results (Limit: {limit}):")
        print(f"Average Top-{args.top_k} Label Consistency: {avg_consistency:.4f}")
        print("(Fraction of retrieved training samples sharing the same label as the test sample)")
        
        # Save results
        save_path = f"attribution_results_{DATASET_NAME}.pt"
        torch.save(results, save_path)
        print(f"Detailed results saved to {save_path}")
        
        # Show one example (Sample Visualization)
        print("\n--- Sample Visualization (First Test Image) ---")
        res = results[0]
        print(f"Test Image Label: {res['test_label']}")
        print(f"Top {args.top_k} influential training samples:")
        for s in res['top_k_samples']:
            print(f"  Train #{s['train_idx']}: Influence={s['influence']:.4f}, Label={s['label']}")

if __name__ == "__main__":
    main()
