import torch
from scipy.interpolate import CubicSpline
import numpy as np
from blocks import geo
from torch.autograd import Variable
def cubicspline(y,n_bins):
    x = np.arange(n_bins)
    cs = CubicSpline(x,torch.log(y+1e-4).cpu())
    c = torch.tensor(cs.c)
    return c
def prob_map_ang(x,cs,device):
    # cs shape: [4,24,length,length]
    # x shape: [length,length]
    # pmap shape: [length,length]
    # bin[i,j] = int(x[i,j])
    # pmap[i,j] = cs[0,bin[i,j],i,j]*(x[i,j]-bin[i,j])**3 + \
    #             cs[1,bin[i,j],i,j]*(x[i,j]-bin[i,j])**2 + \
    #             cs[2,bin[i,j],i,j]*(x[i,j]-bin[i,j])**1 + \
    #             cs[3,bin[i,j],i,j]*(x[i,j]-bin[i,j])**0 
    bins = torch.arange(1,25).to(device)
    length = x.shape[0]
    b = x.detach().to(torch.long)
    index = torch.bucketize(b,bins)
    index = torch.nn.functional.one_hot(index,24).to(torch.bool).permute(2,0,1)
    cs = cs[:,index].view(4,length,length)
    pmap = torch.exp(cs[0]*(x-b)**3 + \
           cs[1]*(x-b)**2 + \
           cs[2]*(x-b)**1 + \
           cs[3])
    return pmap
def prob_map_bb(x,cs,device):
    # cs shape: [4,23,length,length]
    # x shape: [length,length]
    # pmap shape: [length,length]
    # bin[i,j] = int(x[i,j])
    # pmap[i,j] = cs[0,bin[i,j],i,j]*(x[i,j]-bin[i,j])**3 + \
    #             cs[1,bin[i,j],i,j]*(x[i,j]-bin[i,j])**2 + \
    #             cs[2,bin[i,j],i,j]*(x[i,j]-bin[i,j])**1 + \
    #             cs[3,bin[i,j],i,j]*(x[i,j]-bin[i,j])**0 
    bins = torch.arange(1,24).to(device)
    length = x.shape[0]
    b = x.detach().to(torch.long)
    index = torch.bucketize(b,bins)
    index = torch.nn.functional.one_hot(index,23).to(torch.bool).permute(1,0)
    cs = cs[:,index].view(4,length)
    pmap = torch.exp(cs[0]*(x-b)**3 + \
           cs[1]*(x-b)**2 + \
           cs[2]*(x-b)**1 + \
           cs[3])
    return pmap
def prob_map_dist(x,cs,device):
    # cs shape: [4,39,length,length]
    # x shape: [length,length]
    # pmap shape: [length,length]
    # bin[i,j] = int(x[i,j])
    # pmap[i,j] = cs[0,bin[i,j],i,j]*(x[i,j]-bin[i,j])**3 + \
    #             cs[1,bin[i,j],i,j]*(x[i,j]-bin[i,j])**2 + \
    #             cs[2,bin[i,j],i,j]*(x[i,j]-bin[i,j])**1 + \
    #             cs[3,bin[i,j],i,j]*(x[i,j]-bin[i,j])**0 
    bins = torch.arange(2,41).to(device)
    length = x.shape[0]
    x = torch.clamp(x,min=0,max=40)
    b = x.detach().to(torch.long)
    index = torch.bucketize(b,bins)
    index = torch.nn.functional.one_hot(index,39).to(torch.bool).permute(2,0,1)
    cs_cut = cs[3,38,:,:]
    cs = cs[:,index].view(4,length,length)
    pmap = torch.exp((cs[0]*(x-b)**3 + \
           cs[1]*(x-b)**2 + \
           cs[2]*(x-b)**1 + \
           cs[3] - cs_cut))
    return pmap
def energy_ang(ang,cs,device):
    # ang shape: [length,length]
    # cs shape: [4,23,Length,Length]
    pmap = prob_map_ang(ang,cs,device)
    energy = -torch.log(pmap+1e-4).sum()
    return energy
def energy_dist(dist,cs,device):
    # ang shape: [length,length]
    # cs shape: [4,39,Length,Length]
    pmap = prob_map_dist(dist,cs,device)
    energy = -torch.log(pmap+1e-4).sum()
    return energy
def energy_bb(bb,cs,device):
    # ang shape: [length,length]
    # cs shape: [4,39,Length,Length]
    pmap = prob_map_bb(bb,cs,device)
    energy = -torch.log(pmap+1e-4).sum()
    return energy
def energy(eta,theta,chi,cs_dist_N,cs_dist_P,cs_dist_C4,cs_ome,cs_lam,cs_eta,cs_theta,seq,device):
    length = eta.shape[0]
    bb,base = geo.rebiuld_nt(eta.cpu(),theta.cpu(),chi.cpu(),seq)
    C4 = bb[0]
    P = bb[1][:-1,:]
    N,_,_,_,_,_ = base
    C4 = C4.to(device)
    P = P.to(device)
    N = N.to(device)
    # calculate energy
    loss = torch.zeros(1).to(device)
    loss.requires_grad = True
    ## rebiuld desc coord
    length = len(seq)
    ## calculate params & energy
    pred_dist_P = torch.linalg.vector_norm(torch.sub(P[None,:,:],P[:,None,:]),dim=-1)
    loss = loss + energy_dist(pred_dist_P,cs_dist_P,device)
    pred_dist_C4 = torch.linalg.vector_norm(C4[None,:,:]-C4[:,None,:],dim=-1)
    loss = loss + energy_dist(pred_dist_C4,cs_dist_C4,device)
    pred_dist_N = torch.linalg.vector_norm(N[None,:,:]-N[:,None,:],dim=-1)
    loss = loss + energy_dist(pred_dist_N,cs_dist_N,device)
    pred_lambda = geo.cal_Lambda(P,C4,N,device) / (torch.pi/12)
    loss = loss + energy_ang(pred_lambda,cs_lam,device)
    pred_omega = geo.cal_Omega(P,C4,N,device) / (torch.pi/12)
    loss = loss + energy_ang(pred_omega,cs_ome,device)
    pred_eta,pred_theta = geo.cal_backbone_ang(C4,P,device)
    pred_eta = pred_eta / (torch.pi/12)
    pred_theta = pred_theta / (torch.pi/12)
    loss = loss + energy_bb(pred_eta,cs_eta,device)
    loss = loss + energy_bb(pred_theta,cs_theta,device)
    return loss
def bound_round(ang):
    while torch.any(ang > 2 * torch.pi):
        ang[ang > 2 * torch.pi] = ang[ang > 2 * torch.pi] - 2 * torch.pi
    while torch.any(ang < 0):
        ang[ang < 0] = ang[ang < 0] + 2 * torch.pi   
    return ang
def refine_energy(eta,theta,chi,dist_N,dist_P,dist_C4,Omega,Lambda,pred_eta,pred_theta,seq,device):
    length = len(seq)
    # calculate prob_map
    cs_dist_N = cubicspline(dist_N,40).to(device)
    cs_dist_P = cubicspline(dist_P,40).to(device)
    cs_dist_C4 = cubicspline(dist_C4,40).to(device)
    cs_ome = cubicspline(Omega,25).to(device)
    cs_lam = cubicspline(Lambda,25).to(device)
    cs_eta = cubicspline(pred_eta,24).to(device)
    cs_theta = cubicspline(pred_theta,24).to(device)
    for i in range(5):
        print('=====epoch:{}====='.format(str(i)))
        delta_eta = (2*torch.rand(length)-1) * 2 * torch.pi / 180
        delta_theta = (2*torch.rand(length)-1) * 2 * torch.pi / 180
        delta_chi = (2*torch.rand(length)-1) * 2 * torch.pi / 180
        eta = eta + delta_eta.to(device)
        theta = theta + delta_theta.to(device)
        chi = chi + delta_chi.to(device)
        eta = Variable(eta)
        eta.requires_grad = True
        theta = Variable(theta)
        theta.requires_grad = True
        chi = Variable(chi)
        chi.requires_grad = True
        optimizer = torch.optim.LBFGS(params=[eta,theta,chi],lr=1,max_iter=2000,history_size=10000,line_search_fn='strong_wolfe') #'strong_wolfe'?
        #optimizer = torch.optim.Adam(params=[eta,theta,chi], lr=1.2)
        def closure():
            optimizer.zero_grad() 
            loss = torch.zeros(1).to(device)
            loss.requires_grad = True
            loss = loss + energy(eta,theta,chi,cs_dist_N,cs_dist_P,cs_dist_C4,cs_ome,cs_lam,cs_eta,cs_theta,seq,device)
            # backward
            loss.backward()
            print(loss)
            return loss
        for _ in range(1):
            optimizer.step(closure=closure)
            eta = eta % (torch.pi*2)
            theta = theta % (torch.pi*2)
            chi = chi % (torch.pi*2)
    print("energy:",energy(eta,theta,chi,cs_dist_N,cs_dist_P,cs_dist_C4,cs_ome,cs_lam,cs_eta,cs_theta,seq,device).item())
    return eta,theta,chi