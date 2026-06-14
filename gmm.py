import time
import torch
import matplotlib.pyplot as plt

# Date generation:
#   N points, K Gaussians, D-dimensional space,
#   the mu's, sigma's and weights can be given or randomly chosen
#   Input: N, K, D, Mu=None, Sigma=None, Pi=None
#   Return: pts, Mu, Sigma, Pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(time.time())
torch.manual_seed(SEED)


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
    def __init__(self, K, D, reg=1e-6, device=DEVICE):
        self.K = K
        self.D = D
        self.reg = reg
        self.device = device

    def _init_params(self, pts, Mu = None):
        self.N = len(pts)
        indices = torch.randint(self.N, (self.K,), device=self.device)
        self.Mu = pts[indices]
        self.Sigma = torch.eye(self.D, device=self.device).repeat(self.K, 1, 1)
        self.Pi = torch.full((self.K,), 1.0/self.K, device=self.device)

    def _log_gaussian(self, X):
        # compute log(normal(x_n | mu_k, sigma_k)) N x K
        Sigma = self.Sigma + torch.eye(self.D, device=self.device) * self.reg
        L = torch.linalg.cholesky(Sigma) # K x D x D
        diff = X.unsqueeze(1) - self.Mu.unsqueeze(0) # N x K x D
        y = torch.linalg.solve_triangular(L, diff.permute(1,2,0),upper=False) # K x D x N
        mahalanobis = y.norm(dim=1).pow(2).T # N x K
        log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (K,)
        log_norm = -0.5 * (self.D * torch.log(torch.tensor(2 * torch.pi)) + log_det)
        return log_norm.unsqueeze(0) - 0.5 * mahalanobis         # (N, K)
    
    def _e_step(self, X):
        log_prob = self._log_gaussian(X) + torch.log(self.Pi).unsqueeze(0)  # (N, K)
        log_norm = torch.logsumexp(log_prob, dim=1, keepdim=True)           # (N, 1)
        log_resp = log_prob - log_norm            
        ll = log_norm.sum()                                                 # Σ_n log p(x_n)
        return log_resp, ll
    
    def _m_step(self, X, log_resp):
        resp = log_resp.exp() # N x K    
        Nk = resp.sum(dim=0) + 1e-10  # K
        self.Mu = resp.T @ X / Nk.unsqueeze(1) # K x D
        diff =  X.unsqueeze(1) - self.Mu.unsqueeze(0) # N x K x D
        self.Sigma = (resp.unsqueeze(2) * diff).permute(1,2,0) @ diff.permute(1,0,2)/Nk.view(-1,1,1)
        self.Pi = Nk/len(X)
        
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
            

pts, Mu, Sigma, Pi, labels = generate_data(10000, 5, 2)
gmm = GMM(5,2).fit(pts, verbose=True)
plt.scatter(pts[:, 0], pts[:, 1], c=labels, s=8)
plt.scatter(gmm.Mu[:, 0], gmm.Mu[:, 1],
            c="black", marker="X", s=200)  
print(gmm.Mu.cpu().numpy())
plt.scatter(Mu[:, 0], Mu[:, 1],
            c="red", marker="^", s=100)  
plt.gca().set_aspect("equal")
plt.show()
