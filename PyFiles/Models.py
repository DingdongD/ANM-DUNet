import sympy
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def toeplitz(c):  # c : 1 * M vector
    # Hermitian Toeplitz
    c[0, 0] = c[0, 0].conj()
    r = c
    c = c.conj()

    # column vector
    r = r.reshape(-1, 1)
    c = c.reshape(-1, 1)

    p, m = r.shape[0], c.shape[0]

    x1_id = torch.arange(-1, -p, -1)
    x1 = torch.zeros((p - 1, 1), dtype=torch.cdouble).to(device)
    for i in range(p - 1):
        x1[i] = r[int(x1_id[i])]

    x = torch.vstack([x1, c])

    id1 = torch.arange(0, m)
    id2 = torch.arange(p - 1, -1, -1)
    id1 = id1.reshape(-1, 1)

    id = id1 + id2
    T = torch.zeros_like(id, dtype=torch.cdouble).to(device)

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            T[i][j] = x[int(id[i][j])]

    return T


def toeplitz_adjoint(A):
    M = np.shape(A)[0]
    T = torch.zeros((M, 1), dtype=torch.cdouble).to(device)
    T[0, 0] = torch.diag(A).sum(dtype=torch.cdouble)

    for i in range(M-1):
        T[i+1, 0] = torch.diag(A, i+1).sum(dtype=torch.cdouble)

    return T
    
    

class RC_layer(nn.Module):
    def __init__(self):
        super(RC_layer, self).__init__()

    def forward(self, Y, rho, Lamda, Theta, M, L):
        X_hat = 1 / (1 + 2 * rho) * (Y + 2 * Lamda[:, 0:M, M:M + L] + 2 * rho * Theta[:, 0:M, M:M + L])

        return X_hat


class AU_layer(nn.Module):
    def __init__(self):
        super(AU_layer, self).__init__()

    def forward(self, rho, tau, Lamda, Theta, M):
        L = Lamda.size(1) - M
        Imatrix = torch.eye(L, dtype=torch.cdouble).to(device)
        # print('Theta', Lamda, 'Imatrix', Imatrix, 'T', Theta)
        W = (1. / rho) * Lamda[:, M:, M:] + Theta[:, M:, M:] * 1. - tau / (2. * rho) * Imatrix
        return W


class TO_layer(nn.Module):
    def __init__(self):
        super(TO_layer, self).__init__()

    def forward(self, rho, tau, Lamda, Theta, M):
        '''
        Imatrix = torch.eye(M).to(device)
        dummyT = - tau / (2. * rho) * Imatrix + Theta[:, 0:M, 0:M] + Lamda[:, 0:M, 0:M] / (rho * 1.)

        uvec = torch.zeros((Lamda.size(0), 1, M), dtype=torch.cdouble).to(device)
        for i in range(M):
            for j in range(i, M):
                uvec[:, 0, i] = uvec[:, 0, i] + dummyT[:, j, j - i]
            uvec[:, 0, i] /= (M - i) * 1.

        # uvec1 = uvec.cpu().clone()
        # uvec1 = uvec1.detach().numpy()
        T = torch.zeros((Lamda.shape[0], M, M), dtype=torch.cdouble).to(device)
        for i in range(Lamda.shape[0]):
            T[i, :, :] = toeplitz(uvec[i])  
        '''
        bound = torch.arange(M, 0, -1).to(device)
        bound = bound.reshape(-1, 1)
        normalizer = 1. / bound

        e1 = torch.zeros((M, 1), dtype=torch.cdouble).to(device)
        e1[0, 0] = 1
        T = torch.zeros((Lamda.shape[0], M, M), dtype=torch.cdouble).to(device)
        for i in range(Lamda.shape[0]):
            
            u = 1 / rho * normalizer * (toeplitz_adjoint(Lamda[i, 0:M, 0:M]) + \
            rho * toeplitz_adjoint(Theta[i, 0:M, 0:M]) - tau / 2 * M * e1)
       
            T[i, :, :] = toeplitz(u)
            
        # T = torch.from_numpy(T).to(device)
        return T


class NO_layer(nn.Module):
    def __init__(self):
        super(NO_layer, self).__init__()

    def forward(self, rho, T, X, W, Lamda):
        X_ = torch.conj(X)
        X_ = X_.transpose(2, 1)

        r1 = torch.cat((T, X), dim=2)
        r2 = torch.cat((X_, W), dim=2)

        StackM = torch.cat((r1, r2), dim=1)
        Theta = - 1 / (rho) * Lamda + StackM

        Theta1 = Theta.detach()
        for i in range(np.shape(Theta1)[0]):
            _, eig_vector = torch.linalg.eigh(Theta1[i])
            min_vec = torch.randn((Theta1.shape[1]), dtype=torch.cdouble).to(device)
            eps_mat = torch.mm(torch.mm(eig_vector, torch.diag(min_vec)), eig_vector.T.conj())
            Theta[i] += eps_mat

        Sita = torch.zeros_like(Theta, dtype=torch.cdouble).to(device)

        for i in range(np.shape(Theta)[0]):
            eig_value, eig_vector = torch.linalg.eigh(Theta[i])  
            eig_idx = torch.nonzero(eig_value > 0)
            V_temp = torch.zeros(Theta1.shape[1], eig_idx.shape[0], dtype=torch.cdouble).to(device)
            E_temp = torch.zeros((eig_idx.shape[0], eig_idx.shape[0]), dtype=torch.cdouble).to(device)

            for j in range(eig_idx.shape[0]):
                V_temp[:, j] = eig_vector[:, int(eig_idx[j])]
                E_temp[j, j] = eig_value[int(eig_idx[j])]

            Sita[i, :, :] = V_temp.mm(E_temp).mm(V_temp.T.conj())

        Sita = (Sita + torch.transpose(Sita, 1, 2).conj()) / 2

        return Sita


class MU_layer(nn.Module):
    def __init__(self):
        super(MU_layer, self).__init__()

    def forward(self, eta, T, X, W, Lamda, Theta):
        X_ = torch.conj(X).permute(0, 2, 1)
        r1 = torch.cat((T, X), dim=2)  # T:M*M X:M*1
        r2 = torch.cat((X_, W), dim=2)  # XH:1*M W:1*1

        StackM = torch.cat((r1, r2), dim=1)
        Lamda = Lamda + eta * (Theta - StackM)
        return Lamda


class AnmNetwork(nn.Module):
    def __init__(self, rho=0.5, eta=0.1, tau=0.5, K=10):
        super(AnmNetwork, self).__init__()

        self.RO = nn.ParameterList([nn.Parameter(torch.Tensor([rho])) for i in range(K)])  
        self.TA = nn.ParameterList([nn.Parameter(torch.Tensor([tau])) for i in range(K)])  
        self.ET = nn.ParameterList([nn.Parameter(torch.Tensor([eta])) for i in range(K)])  
        self.max_iter = K  # µü´ú´ÎÊý
        '''
        self.rc_layers = nn.ModuleList([RC_layer() for i in range(K)])
        self.au_layers = nn.ModuleList([AU_layer() for i in range(K)])
        self.to_layers = nn.ModuleList([TO_layer() for i in range(K)])
        self.no_layers = nn.ModuleList([NO_layer() for i in range(K)])
        self.mu_layers = nn.ModuleList([MU_layer() for i in range(K)])
        '''
        self.rc_layers = RC_layer()
        self.au_layers = AU_layer()
        self.to_layers = TO_layer()
        self.no_layers = NO_layer()
        self.mu_layers = MU_layer()

    def forward(self, Y):
        batchs, M, L = Y.size(0), Y.size(1), Y.size(2)  # batch * M * snap

        Theta = torch.zeros(batchs, M + L, M + L, dtype=torch.cdouble).to(device)
        Lamda = torch.zeros(batchs, M + L, M + L, dtype=torch.cdouble).to(device)


        for i in range(self.max_iter):
            rho = self.RO[i]
            tau = self.TA[i]
            eta = self.ET[i]

            X_new = self.rc_layers(Y, rho, Lamda, Theta, M, L)
            W_new = self.au_layers(rho, tau, Lamda, Theta, M)
            T_new = self.to_layers(rho, tau, Lamda, Theta, M)
            Theta = self.no_layers(rho, T_new, X_new, W_new, Lamda)
            Lamda = self.mu_layers(eta, T_new, X_new, W_new, Lamda, Theta)

        uvec = torch.zeros((Lamda.size(0), 1, M), dtype=torch.cdouble).to(device)

        for i in range(M):
            for j in range(i, M):
                uvec[:, 0, i] = uvec[:, 0, i] + T_new[:, j, j - i]
            uvec[:, 0, i] /= (M - i) * 1.

        # return spectrum
        uvec = uvec.squeeze(1)

        return T_new, uvec


class CMSELoss(nn.Module):
    def __init__(self, reduction="sum"):
        super(CMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        if self.reduction == "mean":
            l1 = (torch.pow(output - target, 2)).mean(dtype=torch.cdouble)
            l2 = torch.pow(target, 2).mean(dtype=torch.cdouble)

            return torch.abs(l1/l2)

        if self.reduction == "sum":
            l1 = (torch.pow(output - target, 2)).sum(dtype=torch.cdouble)
            l2 = torch.pow(target, 2).sum(dtype=torch.cdouble)
            return torch.abs(l1/l2)



