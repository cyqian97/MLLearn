import time
import torch
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIMENSION = 2
SEED = int(time.time())  # 0
GENERATOR = torch.Generator(device=DEVICE).manual_seed(SEED)


def generate_data(n_centers, n_points, std,
                  dimension=DIMENSION, device=DEVICE, generator=GENERATOR):
    # sample ceFnters
    centers = torch.rand(n_centers, dimension,
                         generator=generator, device=device)
    pts = torch.cat([
        center + std * torch.randn(n_points, dimension,
                                   generator=generator, device=device)
        for center in centers
    ], dim=0)
    return pts


def kmeans_metropolis_init(pts, K, step=10, centroids=None, assign=None, empty_idx=None, generator=GENERATOR):
    # Step 0: uniformly choose a point
    # Step 1: for i in K:
    #   for _ in steps:
    #       propose next uniformaly
    #       compute transition probability (compute individual dist)
    #       decide whether to move
    #   Set centroid


    # To handle initialization and re-initialization
    # For initialization: set centroid[0], empty_idx = [1, 2, ...], 
    # For re-initialization: use assign and empty_idx directly
    
    # Avoid all O(N) operations
    device = pts.device
    if centroids is None:
        centroids = torch.zeros((K, pts.shape[1]), device=device)
        idx = torch.randint(len(pts), (1,), device=device,
                            generator=generator)
        centroids[0] = pts[idx]
        empty_idx = torch.arange(1, K)
        
    live_mask = torch.ones((len(centroids),), dtype=bool, device=device)
    live_mask[e]

    # to accelerate the sample, generate random number all at once
    thresholds = torch.rand((K, step), device=device, generator=generator)
    next_indices = torch.randint(
        len(pts), (K, step), device=device, generator=generator)

    for i in range(K-1):
        idx = torch.randint(len(pts), (1,), device=device,
                            generator=generator)
        for j in range(step):
            next_idx = next_indices[i, j]
            dist_sq_next = torch.min((centroids[live_mask]-pts[next_idx]).norm(dim=1).pow(2)).item()
            dist_sq = torch.min((centroids[live_mask]-pts[idx]).norm(dim=1).pow(2)).item()
            alpha = dist_sq_next/dist_sq
            if thresholds[i, j] < alpha:
                idx = next_idx
        centroids[empty_idx[i]] = pts[idx]
        live_mask[empty_idx[i]] = True
    return centroids


def kmeans_plusplus_init(pts, K, centroids=None, assign=None, empty_idx=None, generator=GENERATOR):
    # Sample p as a centroid with probality dist(p,pts)/sum(p,pts)
    assert K > 0, f"k must be positive, got {k}"
    assert not empty_idx or (K == len(
        empty_idx)), f"K={k} must be match len(empty_idx)={len(empty_idx)}"
    device = pts.device
    if not centroids:
        # intialize the first centroid uniformaly from the points
        centroids = torch.zeros(K, pts.shape[1], device=device)
        centroids[0, :] = pts[torch.randint(
            len(pts), (1,), device=device, generator=generator), :]
        assign = torch.zeros((len(pts),), dtype=torch.long,
                             device=device)  # N x 1
        empty_idx = torch.arange(1, K, device=device)

    min_dist_sq = (pts - centroids[assign]).norm(dim=1).pow(2)

    for i in empty_idx.tolist():
        total = min_dist_sq.sum()
        if total > 0:
            probs = min_dist_sq/min_dist_sq.sum()
        else:
            raise ValueError("min_dist_sq.sum() must be greater than 0!")
        next_id = torch.multinomial(probs, 1, generator=generator)
        centroids[i] = pts[next_id.item()]
        dist_sq = (pts - centroids[i]).norm(dim=1).pow(2)
        min_dist_sq = torch.minimum(dist_sq, min_dist_sq)
    return centroids


def metropolis_init(pts, K, generator=None):
    pass


def kmeans(pts, K, max_iters=1000, tol=1e-5, init_method=None,
           device=DEVICE, dimension=DIMENSION, generator=GENERATOR):
    if not init_method:
        centroids = torch.rand(
            K, dimension, device=device, generator=generator)
    else:
        centroids = init_method(pts, K)

    for _ in range(max_iters):
        # Step 1: assign points to centroids
        dists = torch.cdist(pts, centroids)
        assign = dists.argmin(dim=1)

        # Step 2: update centroids
        new_centroids = centroids.clone()

        # # Method 1: normal loop
        # for i in range(K):
        #     mask = assign == i
        #     if mask.any():
        #         centroid = pts[mask].mean(dim=0)
        #         new_centroids[i] = centroid
        #     else:
        #         new_centroids[i] = torch.rand(
        #             1, dimension, device=device, generator=generator)[0]

        # Method 2
        sums = torch.zeros_like(centroids).index_add_(0, assign, pts)  # (K,d)
        counts = torch.bincount(assign, minlength=K).unsqueeze(1)  # (K,1)
        new_centroids = sums/counts.clamp(min=1)
        is_empty = counts.squeeze(1) == 0
        num_empty = (is_empty).sum()
        empty_idx = (is_empty == 0).nonzero().squeeze()
        if num_empty > 0:
            if init_method:
                init_method(pts, len(empty_idx),
                            new_centroids, assign, empty_idx)
            else:
                # use the farest points
                min_dists = (pts - centroids[assign]).norm(dim=1)
                indices = min_dists.topk(num_empty).indices
                new_centroids[counts.squeeze(1) == 0] = pts[indices]

        if (tol > (new_centroids-centroids).norm(dim=1)).all():
            break
        else:
            centroids = new_centroids

    centroids = new_centroids
    min_dists_mean = (pts-centroids[assign]).norm(dim=1).pow(2).mean().sqrt()
    return centroids, assign, min_dists_mean


pts = generate_data(5, 500, 0.05)
_, _, _ = kmeans(pts, 5, init_method=kmeans_plusplus_init)

# normal initialization
print("normal initialization")
start = time.perf_counter()
cost = 0.
for _ in range(10):
    centroids, assign, min_dists_mean = kmeans(pts, 5)
    cost += min_dists_mean
elapsed = time.perf_counter() - start
print(f"elapsed cpu: {elapsed/10.:.4f} s")
print(f"cost: {cost/10.:.4f}")

# ++ initialization
print("++ initialization")
start = time.perf_counter()
cost = 0.
for _ in range(10):
    centroids, assign, min_dists_mean = kmeans(
        pts, 5, init_method=kmeans_plusplus_init)
    cost += min_dists_mean
elapsed = time.perf_counter() - start
print(f"elapsed cpu: {elapsed/10.:.4f} s")
print(f"cost: {cost/10.:.4f}")

# metropolis initialization
print("metropolis initialization")
start = time.perf_counter()
cost = 0.
for _ in range(10):
    centroids, assign, min_dists_mean = kmeans(
        pts, 5, init_method=kmeans_metropolis_init)
    cost += min_dists_mean
elapsed = time.perf_counter() - start
print(f"elapsed cpu: {elapsed/10.:.4f} s")
print(f"cost: {cost/10.:.4f}")


pts = pts.cpu().numpy()
lbl = assign.cpu().numpy()
centroids = centroids.cpu().numpy()
plt.scatter(pts[:, 0], pts[:, 1], c=lbl, s=8, cmap="tab10")
plt.scatter(centroids[:, 0], centroids[:, 1],
            c="black", marker="X", s=200)  # centroids
plt.gca().set_aspect("equal")
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
