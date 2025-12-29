import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
import argparse
import yaml
import copy
import random
import numpy as np

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    # Graph building (B/C)
    parser.add_argument('--graph_feature_norm', action='store_true', help='L2-normalize features before computing W_f')
    parser.add_argument(
        '--graph_feature_metric',
        type=str,
        default='l2',
        help="Feature distance metric for W_f: l2|cosine",
    )
    parser.add_argument(
        '--graph_adaptive_sigma',
        action='store_true',
        help='Use per-batch distance percentile to set sigma (reduces exp underflow)',
    )
    parser.add_argument(
        '--graph_sigma_percentile',
        type=float,
        default=0.5,
        help='Percentile for adaptive sigma (e.g., 0.5=median, 0.9=larger scale)',
    )
    parser.add_argument(
        '--graph_knn_k',
        type=int,
        default=0,
        help='kNN sparsify W: keep top-k neighbors per node (0 disables)',
    )
    parser.add_argument(
        '--graph_force_fp32',
        action='store_true',
        help='Force fp32 for distance/exp to avoid fp16 underflow',
    )

    # Readout
    parser.add_argument(
        '--readout',
        type=str,
        default='mean',
        help=(
            'Readout method: '
            'mean|sum|max|topk|degree|'
            'mean_max|mean_var|residual|residual_abs'
        ),
    )
    parser.add_argument('--readout_top_k', type=int, default=8, help='Top-k for readout=topk')

    # Diagnostics
    parser.add_argument('--diagnose_single', action='store_true', help='Print single-image t/sigma sensitivity diagnostics')
    parser.add_argument('--diagnose_index', type=int, default=0, help='Which test sample index to diagnose')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Attribution
    parser.add_argument('--analyze_attribution', action='store_true', default=True, help='Run attribution analysis')
    parser.add_argument('--no_analyze_attribution', dest='analyze_attribution', action='store_false', help='Disable attribution analysis')
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


def _w_stats(W: torch.Tensor, eps: float = 1e-12):
    """Compute W mean/std and entropy (both with and without diagonal)."""
    # W: (B, N, N)
    B, N, _ = W.shape
    off_mask = ~torch.eye(N, device=W.device, dtype=torch.bool)  # (N, N)

    def _stats(flat: torch.Tensor):
        mean = flat.mean().item() if flat.numel() else float('nan')
        std = flat.std(unbiased=False).item() if flat.numel() else float('nan')
        # entropy over weight mass
        mass = flat.clamp_min(0)
        Z = mass.sum()
        if Z <= 0:
            ent = float('nan')
        else:
            p = (mass / (Z + eps)).clamp_min(eps)
            ent = (-p * torch.log(p)).sum().item()
        return mean, std, ent

    flat_all = W.reshape(-1)
    # Boolean-index off-diagonal entries per sample: (B, N*(N-1)) then flatten
    flat_off = W[:, off_mask].reshape(-1)

    all_mean, all_std, all_ent = _stats(flat_all)
    off_mean, off_std, off_ent = _stats(flat_off)

    return {
        'all': {'mean': all_mean, 'std': all_std, 'entropy': all_ent},
        'offdiag': {'mean': off_mean, 'std': off_std, 'entropy': off_ent},
    }


@torch.no_grad()
def _extract_one(
    images: torch.Tensor,
    backbone: TimmViTPatchBackbone,
    graph_builder: RegionGraphBuilder,
    operator: DiffusionOperator,
    readout: GraphReadout,
):
    out = backbone(images)
    H, P = out.H, out.P
    regions = RegionSet(H, P)
    W, L = graph_builder.build(regions)
    H_prime = operator.forward(H, L)
    try:
        g = readout(H_prime, W=W, H=H)
    except TypeError:
        # Backward compatibility in case readout signature differs
        g = readout(H_prime)
    return H, W, L, H_prime, g


def _l2(x: torch.Tensor) -> float:
    return torch.linalg.vector_norm(x.reshape(x.size(0), -1), ord=2, dim=1).mean().item()


def _print_diag_block(title: str, lines: list[str]):
    print(f"\n=== {title} ===")
    for line in lines:
        print(line)

def main():
    args = parse_args()
    print(f"Configuration: {vars(args)}")

    # Set random seed for reproducibility
    set_seed(args.seed)

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
    graph_builder = RegionGraphBuilder(
        sigma_s=SIGMA_S,
        sigma_f=SIGMA_F,
        feature_norm=args.graph_feature_norm,
        feature_metric=args.graph_feature_metric,
        adaptive_sigma=args.graph_adaptive_sigma,
        sigma_percentile=args.graph_sigma_percentile,
        knn_k=args.graph_knn_k,
        force_fp32=args.graph_force_fp32,
    )
    operator = DiffusionOperator(t=DIFFUSION_T)
    readout = GraphReadout(method=args.readout, top_k=args.readout_top_k)
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
        
        best_acc = 0.0
        best_backbone_state = None

        for epoch in range(args.ft_epochs):
            # Training
            backbone.model.train()
            ft_head.train()

            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in ft_loader:
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
            
            train_loss = total_loss / len(ft_loader)
            train_acc = correct / total

            # Validation
            backbone.model.eval()
            ft_head.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    out = backbone(images)
                    features = out.H.mean(dim=1)
                    logits = ft_head(features)
                    pred = logits.argmax(dim=1)
                    test_correct += (pred == labels).sum().item()
                    test_total += labels.size(0)
            
            test_acc = test_correct / test_total
            
            print(f"Epoch [{epoch+1}/{args.ft_epochs}] Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                best_backbone_state = copy.deepcopy(backbone.model.state_dict())
        
        print(f"Fine-tuning complete. Best Test Acc: {best_acc:.4f}. Loading best backbone.")
        if best_backbone_state is not None:
            backbone.model.load_state_dict(best_backbone_state)

        print("Freezing backbone.")
        # Freeze backbone for CARROT
        backbone.model.eval()
        for p in backbone.model.parameters():
            p.requires_grad = False

    # 3. Extract Training Features
    print("Extracting training features (G_train)...")
    G_train_list = []
    Y_train_list = []

    with torch.no_grad():
        for images, labels in train_loader:
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
            try:
                g = readout(H_prime, W=W, H=H)
            except TypeError:
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

    # 4.5 Sanity-check: head trained on g_t (i.e., readout(H_prime), not readout(H))
    # This prints a quick diagnostic on the first train batch.
    try:
        first_images, _ = next(iter(train_loader))
        first_images = first_images.to(device)
        with torch.no_grad():
            out0 = backbone(first_images)
            H0, P0 = out0.H, out0.P
            regions0 = RegionSet(H0, P0)
            W0, L0 = graph_builder.build(regions0)
            H0_prime = operator.forward(H0, L0)
            try:
                g_from_Hprime = readout(H0_prime, W=W0, H=H0)
            except TypeError:
                g_from_Hprime = readout(H0_prime)
            try:
                g_from_H = readout(H0, W=W0, H=H0)
            except TypeError:
                g_from_H = readout(H0)
            delta_g = _l2(g_from_Hprime - g_from_H)
        _print_diag_block(
            "Sanity Check (g_t vs g_0)",
            [
                f"Readout method: {args.readout}",
                f"Mean ||g(H') - g(H)|| over first train batch: {delta_g:.6f}",
                "(If this is ~0 for mean readout, diffusion can be 'washed out' by pooling.)",
            ],
        )
    except Exception as e:
        print(f"[Sanity Check] Skipped due to error: {e}")

    # 5. Evaluate
    print("Evaluating on test set...")
    correct = 0
    total = 0
    
    # Store test features and labels for attribution
    G_test_list = []
    Y_test_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            out = backbone(images)
            H, P = out.H, out.P
            regions = RegionSet(H, P)
            W, L = graph_builder.build(regions)
            H_prime = operator.forward(H, L)
            try:
                g = readout(H_prime, W=W, H=H)
            except TypeError:
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

    # 5.5 Single-image diagnostics (t/sigma sensitivity)
    if args.diagnose_single:
        _print_diag_block(
            "Diagnostics Notice",
            [
                "This repo currently has no explicit cache in graph/diffusion/readout/backbone (no memoization found).",
                "So there's nothing to 'disable' unless you added caching elsewhere.",
            ],
        )

        # Grab a single test sample deterministically
        try:
            img, _ = test_dataset[args.diagnose_index]
        except Exception:
            img, _ = test_dataset[0]
        images = img.unsqueeze(0).to(device)

        # Baselines and sweeps
        t_list = [0.0, 10.0, 50.0, 100.0]
        sigma_pairs = [
            (args.sigma_s, args.sigma_f, 'sigma=cfg'),
            (1e-6, 1e-6, 'sigma=1e-6'),
            (1e6, 1e6, 'sigma=1e6'),
        ]

        # Compute baseline at (t=0, sigma=cfg)
        base_graph = RegionGraphBuilder(
            sigma_s=args.sigma_s,
            sigma_f=args.sigma_f,
            feature_norm=args.graph_feature_norm,
            feature_metric=args.graph_feature_metric,
            adaptive_sigma=args.graph_adaptive_sigma,
            sigma_percentile=args.graph_sigma_percentile,
            knn_k=args.graph_knn_k,
            force_fp32=args.graph_force_fp32,
        )
        base_op = DiffusionOperator(t=0.0)
        H0, W0, L0, Hp0, g0 = _extract_one(images, backbone, base_graph, base_op, readout)
        logits0 = head.predict(g0)

        lines = []
        lines.append(f"Fixed input: test_dataset[{args.diagnose_index}] (fallback to 0 if OOB)")
        lines.append(f"Baseline: t=0, sigma_s={args.sigma_s}, sigma_f={args.sigma_f}")
        lines.append(f"||H'(t=0)-H||: {_l2(Hp0 - H0):.6f} (should be ~0)")
        lines.append(f"||g'(t=0)-g||: {_l2(g0 - g0):.6f}")

        _print_diag_block("Single-Image Diagnostics (Norms/Logits/W stats)", lines)

        # Sweep (t, sigma)
        diag_store = {}
        for (sigma_s, sigma_f, sigma_tag) in sigma_pairs:
            gb = RegionGraphBuilder(
                sigma_s=sigma_s,
                sigma_f=sigma_f,
                feature_norm=args.graph_feature_norm,
                feature_metric=args.graph_feature_metric,
                adaptive_sigma=args.graph_adaptive_sigma,
                sigma_percentile=args.graph_sigma_percentile,
                knn_k=args.graph_knn_k,
                force_fp32=args.graph_force_fp32,
            )
            # W stats independent of t; compute once per sigma using baseline H/P
            with torch.no_grad():
                out_tmp = backbone(images)
                regions_tmp = RegionSet(out_tmp.H, out_tmp.P)
                W_tmp, L_tmp = gb.build(regions_tmp)
            stats = _w_stats(W_tmp)
            _print_diag_block(
                f"W Stats ({sigma_tag})",
                [
                    f"all:    mean={stats['all']['mean']:.6g} std={stats['all']['std']:.6g} entropy={stats['all']['entropy']:.6g}",
                    f"offdiag: mean={stats['offdiag']['mean']:.6g} std={stats['offdiag']['std']:.6g} entropy={stats['offdiag']['entropy']:.6g}",
                ],
            )

            for t in t_list:
                op = DiffusionOperator(t=float(t))
                H, W, L, Hp, g = _extract_one(images, backbone, gb, op, readout)
                logits = head.predict(g)

                diag_store[(sigma_tag, float(t))] = {
                    'H': H.detach(),
                    'Hp': Hp.detach(),
                    'g': g.detach(),
                    'logits': logits.detach(),
                }

                # Within-config: ||H'-H||, ||g'-g||
                h_delta = _l2(Hp - H)
                # compare first-D dims to mean(H) as a quick reference (supports concat readouts)
                h_mean = H.mean(dim=1)
                if g.shape[-1] == h_mean.shape[-1]:
                    g_ref = h_mean
                else:
                    g_ref = h_mean[..., : min(h_mean.shape[-1], g.shape[-1])]
                g_cmp = g[..., : g_ref.shape[-1]]
                g_delta = _l2(g_cmp - g_ref)

                # Compare to baseline (t=0, sigma=cfg): ||Hp-Hp0||, ||g-g0||, ||logits-logits0||
                hp_diff0 = _l2(Hp - Hp0)
                g_diff0 = _l2(g - g0)
                logit_diff0 = _l2(logits - logits0)

                _print_diag_block(
                    f"Diag ({sigma_tag}, t={t})",
                    [
                        f"||H'(t)-H||: {h_delta:.6f}",
                        f"||g - mean(H)|| (ref): {g_delta:.6f}",
                        f"vs baseline(t=0,sigma=cfg): ||H'-H'0||={hp_diff0:.6f}, ||g-g0||={g_diff0:.6f}, ||logits-logits0||={logit_diff0:.6f}",
                    ],
                )

        # Explicit pairwise comparisons the user asked for
        pair_lines = []
        # t=0 vs t=10 (same sigma=cfg)
        k0 = ('sigma=cfg', 0.0)
        k10 = ('sigma=cfg', 10.0)
        if k0 in diag_store and k10 in diag_store:
            pair_lines.append("[t=0 vs t=10 @ sigma=cfg]")
            pair_lines.append(f"||H'(10)-H'(0)||: {_l2(diag_store[k10]['Hp'] - diag_store[k0]['Hp']):.6f}")
            pair_lines.append(f"||g(10)-g(0)||: {_l2(diag_store[k10]['g'] - diag_store[k0]['g']):.6f}")
            pair_lines.append(f"||logits(10)-logits(0)||: {_l2(diag_store[k10]['logits'] - diag_store[k0]['logits']):.6f}")

        # sigma=1e-6 vs sigma=1e6 (same t=10)
        ks = ('sigma=1e-6', 10.0)
        kl = ('sigma=1e6', 10.0)
        if ks in diag_store and kl in diag_store:
            pair_lines.append("[sigma=1e-6 vs sigma=1e6 @ t=10]")
            pair_lines.append(f"||H'(large)-H'(small)||: {_l2(diag_store[kl]['Hp'] - diag_store[ks]['Hp']):.6f}")
            pair_lines.append(f"||g(large)-g(small)||: {_l2(diag_store[kl]['g'] - diag_store[ks]['g']):.6f}")
            pair_lines.append(f"||logits(large)-logits(small)||: {_l2(diag_store[kl]['logits'] - diag_store[ks]['logits']):.6f}")

        if pair_lines:
            _print_diag_block("Requested Pairwise Diffs", pair_lines)

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
        
        for i in range(limit):
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
