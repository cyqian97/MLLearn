import torch
import time
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(time.time())
torch.manual_seed(SEED)

# def generate_data(n_per_cluster=300, seed=0):
#     """Generate synthetic 2D data from 3 Gaussians."""
#     torch.manual_seed(seed)
#     means = torch.tensor([[0.0, 0.0], [5.0, 5.0], [0.0, 6.0]])
#     # distinct covariances per cluster
#     covs = torch.stack([
#         torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
#         torch.tensor([[1.0, -0.7], [-0.7, 1.0]]),
#         torch.tensor([[0.6, 0.0], [0.0, 1.5]]),
#     ])
#     samples = []
#     for mu, cov in zip(means, covs):
#         L = torch.linalg.cholesky(cov)              # cov = L Lᵀ
#         z = torch.randn(n_per_cluster, 2)
#         samples.append(z @ L.T + mu)                # transform standard normal
#     X = torch.cat(samples, dim=0)
#     return X[torch.randperm(X.shape[0])]            # shuffle

def generate_data(N, K, D, Mu=None, Sigma=None, Pi=None, device=DEVICE):
    # Step 1: randomly choose Mu, Sigma, Pi if not given
    # Currently the sigma's are all diagonal
    if Mu == None:
        Mu = torch.rand((K, D), device=device)
        Sigma = torch.rand((K, D), device=device)*0.3
        Pi = torch.rand((K,), device=device)
        Pi = Pi/Pi.sum()

    # Step 2: generate sample for each gaussian
    samples = []
    labels = []
    for k in range(K):
        samples.append(Mu[k] + torch.randn((int(N*Pi[k]), D),
                       device=device) * Sigma[k])
        labels.append(torch.full((int(N*Pi[k]),), k))
    pts = torch.cat(samples)
    labels = torch.cat(labels)
    return pts, Mu, Sigma, Pi, labels

class GMM:
    def __init__(self, n_components, n_features, reg=1e-6, device="cpu"):
        self.K = n_components
        self.D = n_features
        self.reg = reg
        self.device = device

    def _init_params(self, X):
        N = X.shape[0]
        # init means at random data points (cheap k-means++ alternative)
        idx = torch.randperm(N, device=self.device)[:self.K]
        self.mu = X[idx].clone()                                  # (K, D)
        self.sigma = torch.eye(self.D, device=self.device).repeat(self.K, 1, 1)  # (K, D, D)
        self.pi = torch.full((self.K,), 1.0 / self.K, device=self.device)        # (K,)

    def _log_gaussian(self, X):
        """Return log N(x_n | mu_k, sigma_k) -> (N, K)."""
        N = X.shape[0]
        # add jitter for numerical stability of Cholesky
        sigma = self.sigma + self.reg * torch.eye(self.D, device=self.device)
        L = torch.linalg.cholesky(sigma)                          # (K, D, D)
        diff = X.unsqueeze(1) - self.mu.unsqueeze(0)              # (N, K, D)
        # solve L y = diff^T  -> Mahalanobis term = ||y||^2
        # reshape diff to (K, D, N) for batched triangular solve
        diff_kdn = diff.permute(1, 2, 0)                          # (K, D, N)
        y = torch.linalg.solve_triangular(L, diff_kdn, upper=False)  # (K, D, N)
        mahalanobis = (y ** 2).sum(dim=1).T                      # (N, K)
        # log|sigma| = 2 * sum(log(diag(L)))
        log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (K,)
        log_norm = -0.5 * (self.D * torch.log(torch.tensor(2 * torch.pi)) + log_det)
        return log_norm.unsqueeze(0) - 0.5 * mahalanobis         # (N, K)

    def _e_step(self, X):
        """Return log-responsibilities (N, K) and log-likelihood (scalar)."""
        log_prob = self._log_gaussian(X) + torch.log(self.pi).unsqueeze(0)  # (N, K)
        log_norm = torch.logsumexp(log_prob, dim=1, keepdim=True)           # (N, 1)
        log_resp = log_prob - log_norm                                      # normalize
        ll = log_norm.sum()                                                 # Σ_n log p(x_n)
        return log_resp, ll

    def _m_step(self, X, log_resp):
        resp = log_resp.exp()                                    # (N, K)
        Nk = resp.sum(dim=0) + 1e-10                             # (K,)
        # means
        self.mu = (resp.T @ X) / Nk.unsqueeze(1)                 # (K, D)
        # covariances
        diff = X.unsqueeze(1) - self.mu.unsqueeze(0)             # (N, K, D)
        weighted = resp.unsqueeze(-1) * diff                    # (N, K, D)
        self.sigma = torch.einsum('nkd,nke->kde', weighted, diff) / Nk.view(-1, 1, 1)
        self.sigma += self.reg * torch.eye(self.D, device=self.device)
        # mixing coefficients
        self.pi = Nk / X.shape[0]

    def fit(self, X, max_iters=100, tol=1e-4, verbose=False):
        X = X.to(self.device)
        self._init_params(X)
        prev_ll = -float("inf")
        for it in range(max_iters):
            log_resp, ll = self._e_step(X)
            self._m_step(X, log_resp)
            if verbose:
                print(f"iter {it:3d}  log-likelihood = {ll.item():.4f}")
            if abs(ll.item() - prev_ll) < tol:
                break
            prev_ll = ll.item()
        return self

    @torch.no_grad()
    def predict(self, X):
        log_resp, _ = self._e_step(X.to(self.device))
        return log_resp.argmax(dim=1)


if __name__ == "__main__":
    # X = generate_data()
    X, Mu, Sigma, Pi, labels = generate_data(1000, 5, 2)
    gmm = GMM(n_components=3, n_features=2).fit(X, verbose=True)
    print("\nMixing coefficients:", gmm.pi)
    print("Means:\n", gmm.mu)
    labels = gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=8)
    plt.scatter(gmm.mu[:, 0], gmm.mu[:, 1],
                c="black", marker="X", s=200)  
    print(gmm.mu.cpu().numpy())
    plt.scatter(Mu[:, 0], Mu[:, 1],
                c="red", marker="^", s=100)  
    plt.gca().set_aspect("equal")
    plt.show()
