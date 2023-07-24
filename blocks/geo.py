import torch
from torch import tensor,cross,cos,sin,outer,eye,zeros

# ===Geo Params===
len_p_c = 3.8761
len_c_p = 3.8705
ang_p_c_p = tensor(1.8299)
ang_c_p_c = tensor(1.8444)
torN = tensor([105.814, 102.751, 104.225, 105.158])
angN = tensor([101.164, 99.153, 99.483, 100.687])
distN = tensor([3.392, 3.398, 3.383, 3.413])
torO4 = tensor([114.0196, 108.4575, 112.4488, 112.2701])
angO4 = tensor([131.5051, 131.0396, 130.9602, 130.8516])
distO4 = tensor([1.4505, 1.4512, 1.4512, 1.4509])
torC1 = tensor([128.6302, 129.5699, 131.9457, 127.7324])
angC1 = tensor([109.8054, 109.7353, 109.7436, 109.7484])
distC1 = tensor([1.4127, 1.4141, 1.4132, 1.4134])
angChi = tensor([126.5274, 118.8795, 126.4582, 117.6994])
distChi = tensor([1.3724, 1.3982, 1.3735, 1.3814])
torAfterChi = tensor([359.3861, 179.8174, 359.8648, 179.7748])
angAfterChi = tensor([159.6609, 89.6304, 161.8754, 88.5634])
distAfterChi = tensor([2.1999, 2.3242, 2.2144, 2.4563])
torC6 = tensor([180.2857, 0.0, 180.2485, 0.0])
angC6 = tensor([63.4597, 0.0, 61.8550, 0.0])
distC6 = tensor([2.3104, 0.0, 2.4477, 0.0])
ntid = ['A', 'C', 'G', 'U', 'X', 'I']

# get_nt_id:
def get_nt_id(nt):
    nid = ntid.index(nt)
    if nid == 4:
        nid = 3
    elif nid == 5:
        nid =2
    return nid

# Rotation along vector u
def rot_with_u(u,theta):
    
    u = u/u.norm()
    x,y,z = u
    I = eye(3)
    c = cos(theta)
    s = sin(theta)
    a = outer(u,u)
    b = tensor([
        [0,-z,y],
        [z,0,-x],
        [-y,x,0]
    ])
    R = c*I+(1-c)*a+s*b
    
    return R

# While get bond-length tor and ang, Calculate the coord. .
def cal_coord(atom_1,atom_2,atom_3,length,tor,ang):
    
    x = atom_2-atom_1
    y = atom_3-atom_1
    z = cross(x,y)
    y = cross(z,x)
    
    ex = x/x.norm()
    ey = y/y.norm()
    ez = z/z.norm()
    
    r1 = rot_with_u(ex,tor)
    ez = rot_with_u(ex,tor)@ez
    r2 = rot_with_u(ez,-ang)
    
    return atom_2-length*(r2@r1@ex)

def rebiuld_backbone(eta,theta):
    length = eta.shape[0]
    # init
    C4 = zeros(length,3)
    P = zeros(length+1,3)
    # cal
    P[0] = len_p_c * tensor([cos(ang_p_c_p),sin(ang_p_c_p),0.0])
    C4[0] = tensor([0.0,0.0,0.0])
    P[1] = len_c_p * tensor([1.0,0.0,0.0])
    for i in range(length-1):
        C4[i+1] = cal_coord(C4[i],P[i+1],P[i],len_c_p,eta[i],ang_c_p_c)
        P[i+2] = cal_coord(P[i+1],C4[i+1],C4[i],len_p_c,theta[i],ang_p_c_p)
    return C4,P

def rebiuld_base(C,P,chi,seq):
    length = C.shape[0]
    # init
    N = zeros(length,3)
    O4 = zeros(length,3)
    C1 = zeros(length,3)
    C2 = zeros(length,3)
    C4 = zeros(length,3)
    C6 = zeros(length,3)
    # cal
    for i in range(length):
        nid = get_nt_id(seq[i])
        ## N
        tor = torN[nid]
        ang = angN[nid]
        l = distN[nid]
        N[i] = cal_coord(C[i],P[i+1],P[i],l,tor,ang)
        ## O4
        tor = torO4[nid]
        ang = angO4[nid]
        l = distO4[nid]
        O4[i] = cal_coord(C[i],P[i+1],P[i],l,tor,ang)
        ## C1
        tor = torC1[nid]
        ang = angC1[nid]
        l = distC1[nid]
        C1[i] = cal_coord(O4[i],C[i],P[i],l,tor,ang)
        if nid == 1 or nid == 3:
            ## C2
            tor = chi[i]
            ang = angChi[nid]
            l = distChi[nid]
            C2[i] = cal_coord(N[i],C1[i],O4[i],l,tor,ang)
            ## C4
            tor = torAfterChi[nid]
            ang = angAfterChi[nid]
            l = distAfterChi[nid]
            C4[i] = cal_coord(C2[i],N[i],C1[i],l,tor,ang)
        else:
            ## C4
            tor = chi[i]
            ang = angChi[nid]
            l = distChi[nid]
            C4[i] = cal_coord(N[i],C1[i],O4[i],l,tor,ang)
            ## C2
            tor = torAfterChi[nid]
            ang = angAfterChi[nid]
            l = distAfterChi[nid]
            C2[i] = cal_coord(C4[i],N[i],C1[i],l,tor,ang)
            ## C6
            tor = torC6[nid]
            ang = angC6[nid]
            l = distC6[nid]
            C6[i] = cal_coord(C2[i],C4[i],N[i],l,tor,ang)
    return N,O4,C1,C2,C4,C6

def rebiuld_nt(eta,theta,chi,seq):
    bb = rebiuld_backbone(eta,theta)
    base = rebiuld_base(bb[0],bb[1],chi,seq)
    return bb,base

def cal_Lambda(C4,C1,N,device):
    
    def Rotation_Lambda(u,theta):
    
        length = u.shape[0]
        u = u/(u.norm(dim=1)[:,None]+1e-6)
        x,y,z = u[:,0],u[:,1],u[:,2]
        I = torch.eye(3).to(device)
        c = torch.cos(theta).to(device)
        s = torch.sin(theta).to(device)
        a = torch.einsum('ix,jy->ixy',u,u)

        b = torch.zeros(length,3,3).to(device)
        b[:,0,1] = -z
        b[:,1,0] = z
        b[:,0,2] = y
        b[:,2,0] = -y
        b[:,1,2] = -x
        b[:,2,1] = x

        R = torch.einsum('ij,xy->ijxy',c,I)
        R = R + torch.einsum('ij,ixy->ijxy',(1-c),a)
        R = R + torch.einsum('ij,ixy->ijxy',s,b)
        return R

    t1 = torch.cross(C1-N,C4-N)
    t2 = torch.cross((C1-N)[:,None],N[:,None]-N[None,:])
    t1 = t1/t1.norm(dim=-1)[:,None]
    t2 = t2 * (1/(t2.norm(dim=-1)[:,:,None]+1e-6))
    
    tor = torch.arccos((1-1e-4)*torch.einsum('ix,ijx->ij',t1,t2))
    R = Rotation_Lambda(C1-N,tor)
    t_val = torch.round(torch.einsum('ijxy,iy->ijx',R,t1)*1e3)/1e3
    tor[(t_val != torch.round(t2*1e3)/1e3).sum(dim=-1).to(torch.bool)] = 2*torch.pi - tor[(t_val != torch.round(t2*1e3)/1e3).sum(dim=-1).to(torch.bool)]
    return 2*torch.pi - tor

def cal_Omega(C4,C1,N,device):
    t = torch.cross((C1-N)[:,None],N[:,None]-N[None,:])
    t = t/(t.norm(dim=-1)[:,:,None]+1e-6)
    def Rotation_Omega(u,theta):
    
        length = u.shape[0]
        u = u/(u.norm(dim=-1)[:,:,None]+1e-6)
        x,y,z = u[:,:,0],u[:,:,1],u[:,:,2]
        I = torch.eye(3).to(device)
        c = torch.cos(theta).to(device)
        s = torch.sin(theta).to(device)
        a = torch.einsum('ijx,ijy->ijxy',u,u)
        b = torch.zeros(length,length,3,3).to(device)
        b[:,:,0,1] = -z
        b[:,:,1,0] = z
        b[:,:,0,2] = y
        b[:,:,2,0] = -y
        b[:,:,1,2] = -x
        b[:,:,2,1] = x

        R = torch.einsum('ij,xy->ijxy',c,I)
        R = R+ torch.einsum('ij,ijxy->ijxy',(1-c),a)
        R = R + torch.einsum('ij,ijxy->ijxy',s,b)
        return R
    tor = torch.arccos((1-1e-4)*torch.einsum('ijx,jix->ij',t,t))
    R = Rotation_Omega(t,tor)
    t_val = torch.round(torch.einsum('ijxy,ijy->ijx',R,t)*1e3)/1e3
    tor[(t_val-t).norm(dim=-1)>=0.01] = 2*torch.pi - tor[(t_val-t).norm(dim=-1)>=0.01]
    return 2*torch.pi - tor

def cal_backbone_ang(C4,P,device):
    def Rotation(u,theta):

        u = u/(u.norm()+1e-6)
        x,y,z = u
        I = torch.eye(3).to(device)
        c = torch.cos(theta).to(device)
        s = torch.sin(theta).to(device)
        a = torch.outer(u,u)
        b = torch.tensor([
            [0,-z,y],
            [z,0,-x],
            [-y,x,0]
        ]).to(device)
        R = c*I+(1-c)*a+s*b

        return R

    def cal_Tor(O,A,B,C):

        t1 = torch.cross(A-O,B-O)
        t2 = torch.cross(A-O,C-O)
        t1 = t1/(t1.norm()+1e-6)
        t2 = t2/(t2.norm()+1e-6)
        tor = torch.arccos((1-1e-6)*torch.dot(t1,t2))
        R = Rotation(A-O,tor)
        t_val = R@t1
        if (t_val-t2).norm()>0.1:
            tor = 2*torch.pi - tor
        return tor
    
    length = C4.shape[0]
    eta = torch.zeros(length).to(device)
    theta = torch.zeros(length).to(device)
    for k in range(length-1):
        eta[k] = cal_Tor(C4[k],P[k+1],P[k],C4[k+1])
    for k in range(length-2):
        theta[k] = cal_Tor(P[k+1],C4[k+1],C4[k],P[k+2])
    return eta,theta