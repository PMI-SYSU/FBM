#%%
import numpy as np

def Phi(h):
    
    phi = np.tanh(h)
    return phi

def Theta(x, a):
    y = (x + np.sqrt(x**2 + a))/2
        
    return y#theta

def P_out(zL, zR, yL, yR, λL, λR, dF, λB, β, a):
    
    Y = np.sign(yL*yR + 1)
    Y = np.expand_dims(np.expand_dims(Y,axis=1),axis=2)
    phi_L = Phi(zL)
    phi_R = Phi(zR)
           
    D = np.sqrt((phi_L-phi_R)**2)
    inp = dF - D**2

    E = (Y/2)*λB*D**2 + ((1-Y)/2)*Theta(inp, a) + (λL/2)*phi_L**2 + (λR/2)*phi_R**2          
    Pout = np.exp(-β*E)#/(np.exp(-β*Epp)+ np.exp(-β*Enn)+ np.exp(-β*Epn)+ np.exp(-β*Enp))
  
    return Pout #[P,N,n]

def Initial_a_v(N,P_Bpp,P_Bnn,P_Fpn,P_Fnp):    
    
    a_iμ_Bpp = np.random.normal(0, 1.0,(N,P_Bpp))
    v_iμ_Bpp = np.abs(np.random.normal(0, 1.0,(N, P_Bpp)))#np.abs(np.random.normal(0, 1.0,(N,P)))#np.ones((N,P))

    a_iμ_Bnn = np.random.normal(0, 1.0,(N,P_Bnn))
    v_iμ_Bnn = np.abs(np.random.normal(0, 1.0,(N,P_Bnn)))
 
    a_iμ_Fpn = np.random.normal(0, 1.0,(N,P_Fpn))
    v_iμ_Fpn = np.abs(np.random.normal(0, 1.0,(N,P_Fpn)))#np.abs(np.random.normal(0, 1.0,(N,P)))#np.ones((N,P))

    a_iμ_Fnp = np.random.normal(0, 1.0,(N,P_Fnp))
    v_iμ_Fnp = np.abs(np.random.normal(0, 1.0,(N,P_Fnp)))
    
    a_iμ_YY = [a_iμ_Bpp, a_iμ_Bnn, a_iμ_Fpn, a_iμ_Fnp]
    v_iμ_YY = [v_iμ_Bpp, v_iμ_Bnn, v_iμ_Fpn, v_iμ_Fnp]
    
    return a_iμ_YY, v_iμ_YY

def Update_omega_V(a_iμ_YY, v_iμ_YY, X_YY):

    XBpp = X_YY[0]#[N,P,k]
    XBnn = X_YY[1]
    XFpn = X_YY[2]
    XFnp = X_YY[3]
    
    a_iμ_Bpp = a_iμ_YY[0]#[N,P]
    a_iμ_Bnn = a_iμ_YY[1]
    a_iμ_Fpn = a_iμ_YY[2]
    a_iμ_Fnp = a_iμ_YY[3]
    
    v_iμ_Bpp = v_iμ_YY[0]
    v_iμ_Bnn = v_iμ_YY[1]
    v_iμ_Fpn = v_iμ_YY[2]
    v_iμ_Fnp = v_iμ_YY[3]
    #X = np.concatenate(np.concatenate(XBpp, XBnn, axis=1), np.concatenate(XFpn, XFnp, axis=1), axis=1) 
    ω_μi_Bpp = (np.einsum('ipk, ip->pk', XBpp, a_iμ_Bpp)- (np.einsum('ipk, ip->ipk', XBpp, a_iμ_Bpp))).transpose(1,0,2)    
    ω_μi_Bnn = (np.einsum('ipk, ip->pk', XBnn, a_iμ_Bnn)- (np.einsum('ipk, ip->ipk', XBnn, a_iμ_Bnn))).transpose(1,0,2)    
    ω_μi_Fpn = (np.einsum('ipk, ip->pk', XFpn, a_iμ_Fpn)- (np.einsum('ipk, ip->ipk', XFpn, a_iμ_Fpn))).transpose(1,0,2)    
    ω_μi_Fnp = (np.einsum('ipk, ip->pk', XFnp, a_iμ_Fnp)- (np.einsum('ipk, ip->ipk', XFnp, a_iμ_Fnp))).transpose(1,0,2)    
  
    XvBpp = np.einsum('ipk, ip->ipk', XBpp, v_iμ_Bpp)
    V_μi_Bpp = (np.einsum('ipk, ipl->pkl', XBpp, XvBpp) - np.einsum('ipk, ipl->ipkl', XBpp, XvBpp)).transpose(1,0,2,3)
    XvBnn = np.einsum('ipk, ip->ipk', XBnn, v_iμ_Bnn)
    V_μi_Bnn = (np.einsum('ipk, ipl->pkl', XBnn, XvBnn) - np.einsum('ipk, ipl->ipkl', XBnn, XvBnn)).transpose(1,0,2,3)
    XvFpn = np.einsum('ipk, ip->ipk', XFpn, v_iμ_Fpn)
    V_μi_Fpn = (np.einsum('ipk, ipl->pkl', XFpn, XvFpn) - np.einsum('ipk, ipl->ipkl', XFpn, XvFpn)).transpose(1,0,2,3)
    XvFnp = np.einsum('ipk, ip->ipk', XFnp, v_iμ_Fnp)
    V_μi_Fnp = (np.einsum('ipk, ipl->pkl', XFnp, XvFnp) - np.einsum('ipk, ipl->ipkl', XFnp, XvFnp)).transpose(1,0,2,3)
    
    ω_YY = [ω_μi_Bpp, ω_μi_Bnn, ω_μi_Fpn, ω_μi_Fnp] 
    V_YY = [V_μi_Bpp, V_μi_Bnn, V_μi_Fpn, V_μi_Fnp]        
    
    return ω_YY, V_YY
                

def zt_rand(P_Bpp,P_Bnn,P_Fpn,P_Fnp,N,n):
    zRt_Bpp = np.random.normal(0, 1, size=(P_Bpp,N,n))
    zLt_Bpp = np.random.normal(0, 1, size=(P_Bpp,N,n))
    zt_Bpp = np.random.normal(0, 1, size=(P_Bpp,N,n))

    zRt_Bnn = np.random.normal(0, 1, size=(P_Bnn, N,n))
    zLt_Bnn = np.random.normal(0, 1, size=(P_Bnn, N,n))
    zt_Bnn = np.random.normal(0, 1, size=(P_Bnn, N,n))
    
    zRt_Fpn = np.random.normal(0, 1, size=(P_Fpn,N,n))
    zLt_Fpn = np.random.normal(0, 1, size=(P_Fpn,N,n))
    zt_Fpn = np.random.normal(0, 1, size=(P_Fpn,N,n))

    zRt_Fnp = np.random.normal(0, 1, size=(P_Fnp,N,n))
    zLt_Fnp = np.random.normal(0, 1, size=(P_Fnp,N,n))
    zt_Fnp = np.random.normal(0, 1, size=(P_Fnp,N,n))
    
    ZT = [zt_Bpp, zt_Bnn, zt_Fpn, zt_Fnp]
    ZLT = [zLt_Bpp, zLt_Bnn, zLt_Fpn, zLt_Fnp]
    ZRT = [zRt_Bpp, zRt_Bnn, zRt_Fpn, zRt_Fnp]
    return ZLT, ZRT
    
def z_multivariate(ω_YY, V_YY, P,N,n):
    
    ω_μi_Bpp = ω_YY[0]   
    ω_μi_Bnn = ω_YY[1]
    ω_μi_Fpn = ω_YY[2]
    ω_μi_Fnp = ω_YY[3]
    
    V_μi_Bpp = V_YY[0]
    V_μi_Bnn = V_YY[1]
    V_μi_Fpn = V_YY[2]
    V_μi_Fnp = V_YY[3]    
    
    P_Bpp = ω_μi_Bpp.shape[0]
    Zt_Bpp = np.zeros((P_Bpp,N, 2,n))
    for p in range(P_Bpp):
        for i in range(N):
            z = np.random.multivariate_normal(ω_μi_Bpp[p,i], V_μi_Bpp[p,i], size=n)
            Zt_Bpp[p,i,:,:] = z.T          
    zL_Bpp = Zt_Bpp[:,:,0,:]
    zR_Bpp = Zt_Bpp[:,:,1,:]

    P_Bnn = ω_μi_Bnn.shape[0]
    Zt_Bnn = np.zeros((P_Bnn,N, 2,n))
    for p in range(P_Bnn):
        for i in range(N):
            z = np.random.multivariate_normal(ω_μi_Bnn[p,i], V_μi_Bnn[p,i], size=n)
            Zt_Bnn[p,i,:,:] = z.T          
    zL_Bnn = Zt_Bnn[:,:,0,:]
    zR_Bnn = Zt_Bnn[:,:,1,:]
    
    P_Fpn = ω_μi_Fpn.shape[0]
    Zt_Fpn = np.zeros((P_Fpn,N, 2,n))
    for p in range(P_Fpn):
        for i in range(N):
            z = np.random.multivariate_normal(ω_μi_Fpn[p,i], V_μi_Fpn[p,i], size=n)
            Zt_Fpn[p,i,:,:] = z.T          
    zL_Fpn = Zt_Fpn[:,:,0,:]
    zR_Fpn = Zt_Fpn[:,:,1,:]

    P_Fnp = ω_μi_Fnp.shape[0]
    Zt_Fnp = np.zeros((P_Fnp,N, 2,n))
    for p in range(P_Fnp):
        for i in range(N):
            z = np.random.multivariate_normal(ω_μi_Fnp[p,i], V_μi_Fnp[p,i], size=n)
            Zt_Fnp[p,i,:,:] = z.T          
    zL_Fnp = Zt_Fnp[:,:,0,:]
    zR_Fnp = Zt_Fnp[:,:,1,:]
    
    ZLT = [zL_Bpp, zL_Bnn, zL_Fpn, zL_Fnp]
    ZRT = [zR_Bpp, zR_Bnn, zR_Fpn, zR_Fnp]
    
    return ZLT,ZRT
            
def z_fun(ω_YY,  V_YY, num, ZLT, ZRT):
    ZL_YY = []
    ZR_YY = []
    CZ_YY = []
    for i in range(len(ω_YY)):
        ω_μin_yy = np.expand_dims(ω_YY[i], axis=3)#[P, N, k, 1]
        V_μi_yy_eigvalue, V_μi_yy_eigvector = np.linalg.eig(V_YY[i])#[P,N,k], [P,N,k,l]
        
        ZRt_yy = np.expand_dims(ZRT[i], axis=2)
        ZLt_yy = np.expand_dims(ZLT[i], axis=2)
        Zt_yy = np.concatenate((ZLt_yy,ZRt_yy), axis=2)#[P,N,k,n]
        
        
        if np.any(V_μi_yy_eigvalue<0):
            print('V_eigvalue < 0')
            break
        
        lambdaZ_yy = np.expand_dims(np.sqrt(V_μi_yy_eigvalue), axis=3)*Zt_yy#Zt
        Cz_yy = np.einsum('pikl,piln->pikn', V_μi_yy_eigvector, lambdaZ_yy)
        Zt_yy_new = ω_μin_yy + Cz_yy
        zL_yy = Zt_yy_new[:,:,0,:]#[P,N,k,n]
        zR_yy = Zt_yy_new[:,:,1,:] 
        
        ZL_YY.append(zL_yy)#[P,N,n]
        ZR_YY.append(zR_yy)#[P,N,n]
        CZ_YY.append(Cz_yy)#[P,N,2,n]
    
    return ZL_YY,ZR_YY #[P,N,n]


def g_out(ω_YY, V_YY, YY, ZL_YY,ZR_YY, λL, λR, dF,λB, β, a):

    Gout = []
    for i in range(len(ω_YY)):
        YL = YY[i][:,0]#[P]
        YR = YY[i][:,1]   
        ZR = np.expand_dims(ZR_YY[i], axis=2)#[P,N,n]
        ZL = np.expand_dims(ZL_YY[i], axis=2)
        Z = np.concatenate((ZL,ZR), axis=2)#[P,N,k,n]
        Pout_yz = np.expand_dims(P_out(ZL_YY[i],ZR_YY[i], YL, YR, λL, λR, dF,λB, β, a), axis=2) #[P,N,1,n]   
 
        ω_n_YY = np.expand_dims(ω_YY[i], axis=3) #[P,N,k,1]   
        V_inv_YY = np.linalg.inv(V_YY[i]) #[P,N,k,l] 
        ZOmegaV_YY = np.einsum('pikl, piln -> pikn', V_inv_YY, Z - ω_n_YY)
        gout = np.sum(Pout_yz*ZOmegaV_YY, axis=3)/np.sum(Pout_yz, axis=3) #[P,N,k]   
        Gout.append(gout)
        
    return Gout

def dg_out(Gout, ω_YY, V_YY, YY, ZL_YY,ZR_YY, λL, λR, dF,λB, β, a):

    DGout =[]
    for i in range(len(ω_YY)):
        YL = YY[i][:,0]#[P]
        YR = YY[i][:,1]        
        ZR = np.expand_dims(ZR_YY[i], axis=2)#[P,N,n]
        ZL = np.expand_dims(ZL_YY[i], axis=2)
        Z = np.concatenate((ZL,ZR), axis=2)#[P,N,k,n]    
        Pout_yz = np.expand_dims(np.expand_dims(P_out(ZL_YY[i],ZR_YY[i], YL, YR, λL, λR, dF,λB, β, a), axis=2), axis=2)#[P,N,1,1,n]   
        ω_n_YY = np.expand_dims(ω_YY[i], axis=3) #[P,N,k,1]   
        V_inv_YY = np.linalg.inv(V_YY[i]) #[P,N,k,l] 
        ZOmegaV_YY = np.einsum('pikl, piln -> pikn', V_inv_YY, Z - ω_n_YY)
        ZOmegaV_ZOmegaV_YY = np.einsum('pikn, piln -> pikln', ZOmegaV_YY, ZOmegaV_YY)  
        gout_gout = np.einsum('pik, pil->pikl', Gout[i], Gout[i])
        dgout = - V_inv_YY + np.sum(ZOmegaV_ZOmegaV_YY*Pout_yz, axis=4)/np.sum(Pout_yz, axis=4) - gout_gout
        DGout.append(dgout)

    return DGout #[P,N,k,l]

def BA(xL, xR, gout, dgout):
    XL = np.expand_dims(xL, axis=2)
    XR = np.expand_dims(xR, axis=2)
    X = np.concatenate((XL,XR), axis=2)
    
    B_μi = np.einsum('ipk, pik->pi', X, gout)
    Xdgout = np.einsum('ipk, pikl->pil', X, dgout)
    A_μi = -np.einsum('pil, ipl->pi', Xdgout, X)
 
    return B_μi, A_μi

def Gamma_gamma(X_YY, Gout, DGout):

    B_i = 0
    A_i = 0
    for i in range(len(X_YY)):
        XLR = X_YY[i]
        
        B_μi = np.einsum('ipk, pik->pi', XLR, Gout[i])
        Xdgout = np.einsum('ipk, pikl->pil', XLR, DGout[i])
        A_μi = -np.einsum('pil, ipl->pi', Xdgout, XLR)
        
        B_i += np.sum(B_μi, axis=0)
        A_i += np.sum(A_μi, axis=0)
    
    Γ_μi_YY = []
    γ_μi_YY = []
     
    Γ_i_YY = []
    γ_i_YY = []  
    
    for i in range(len(X_YY)):
        XLR = X_YY[i]
        B_μi = np.einsum('ipk, pik->pi', XLR, Gout[i])
        Xdgout = np.einsum('ipk, pikl->pil', XLR, DGout[i])
        A_μi = -np.einsum('pil, ipl->pi', Xdgout, XLR)
        
        Γ_μi = A_i-A_μi
        γ_μi = B_i-B_μi
     
        Γ_i = np.sum(A_μi, axis=0)
        γ_i = np.sum(B_μi, axis=0)   
        
        Γ_μi_YY.append(Γ_μi)
        γ_μi_YY.append(γ_μi)
        Γ_i_YY.append(Γ_i)
        γ_i_YY.append(γ_i)
        
         
    return γ_μi_YY, Γ_μi_YY, γ_i_YY, Γ_i_YY
            

def av_iμ(γ_μi_YY, Γ_μi_YY, β,λw):
    a_iμ_YY = []
    v_iμ_YY = []   
    for i in range(len(γ_μi_YY)):
        v_iμ = 1/((β*λw + Γ_μi_YY[i]).T)  
        a_iμ = (γ_μi_YY[i]/(β*λw + Γ_μi_YY[i])).T  
    
        a_iμ_YY.append(a_iμ)          
        v_iμ_YY.append(v_iμ)          
     
    return a_iμ_YY, v_iμ_YY

def av(γ_i_YY, Γ_μi_YY, β,λw):
    
    γ_i_Bpp = γ_i_YY[0]
    γ_i_Bnn = γ_i_YY[1]
    γ_i_Fpn = γ_i_YY[2]
    γ_i_Fnp = γ_i_YY[3]    
    
    Γ_i_Bpp = Γ_μi_YY[0]
    Γ_i_Bnn = Γ_μi_YY[1]
    Γ_i_Fpn = Γ_μi_YY[2]
    Γ_i_Fnp = Γ_μi_YY[3]
    
    γ_i = γ_i_Bpp + γ_i_Bnn + γ_i_Fpn + γ_i_Fnp
    Γ_i = Γ_i_Bpp + Γ_i_Bnn + Γ_i_Fpn + Γ_i_Fnp
    
    v_i = (1/(β*λw + Γ_i)).T  
    a_i = (γ_i/(β*λw + Γ_i)).T    
    
    return a_i, v_i

def L(a_i,v_i, YY, X_YY, λL, λR,dF, λB,ρF,ρB, mX,σX, N,a,n):
    
    XBpp = X_YY[0]#[N,P,k]
    XBnn = X_YY[1]
    XFpn = X_YY[2]
    XFnp = X_YY[3]

    X = np.concatenate((np.concatenate((XBpp, XBnn), axis=1), np.concatenate((XFpn, XFnp), axis=1)), axis=1) 

    YBpp = YY[0]#[P,k]
    YBnn = YY[1]
    YFpn = YY[2]
    YFnp = YY[3]
    Y = np.concatenate((np.concatenate((YBpp, YBnn), axis=0), np.concatenate((YFpn, YFnp), axis=0)), axis=0) 

    
    
    Y = np.sign(Y[:,0]*Y[:,1] + 1)
    xL = X[:,:,0]
    xR = X[:,:,1]
    
    t = np.random.normal(0,1, size=(N,n))
    W = np.expand_dims(a_i,axis=1) + np.einsum('N, Nn->Nn',np.sqrt(v_i),t) 
    zL = np.einsum('Nn, NP->nP', W, xL)
    zR = np.einsum('Nn, NP->nP', W, xR)
    phi_L = Phi(zL)
    phi_R = Phi(zR)  
    
    D = np.sqrt((phi_L-phi_R)**2)
    inp = dF - D**2
    
    L_bays = np.mean((Y/2)*λB*D**2 + ((1-Y)/2)*Theta(inp, a) + (λL/2)*phi_L**2 + (λR/2)*phi_R**2)
    zL_op = np.einsum('N, NP->P', a_i, xL)
    zR_op = np.einsum('N, NP->P', a_i, xR)
    
    phi_L = Phi(zL_op)
    phi_R = Phi(zR_op)
            
    D = np.sqrt((phi_L-phi_R)**2)
    inp = dF - D**2      

    L_op = np.mean((Y/2)*λB*D**2 + ((1-Y)/2)*Theta(inp, a) + (λL/2)*phi_L**2 + (λR/2)*phi_R**2)
 
    Pt=2000
    bs = Pt
    yLt = 2*np.random.randint(0,2, size = (Pt,))-1
    xL_m = yLt*np.ones((N, Pt))*mX
    xL_r = np.random.normal(loc=0, scale=σX, size=(N, Pt))
    xLt = xL_m + xL_r
    
    yf = -np.ones((int(bs*ρF)))
    yb = np.ones((int(bs*ρB)))
    y_batch = np.hstack((yf,yb))
    yRt = y_batch.repeat(int(Pt/bs), axis=0)*yLt
    xR_m = yRt*np.ones((N, Pt))*mX
    xR_r = np.random.normal(loc=0, scale=σX, size=(N, Pt))
    xRt = xR_m + xR_r
    
    Y = np.sign(yLt*yRt + 1)
    
    
    zLt = np.einsum('Nn, NP->nP', W, xLt)
    zRt = np.einsum('Nn, NP->nP', W, xRt)
    
    phi_L = Phi(zLt)
    phi_R = Phi(zRt)  
    
    D = np.sqrt((phi_L-phi_R)**2)
    inp = dF - D**2
    
    Lt_bays =  np.mean((Y/2)*λB*D**2 + ((1-Y)/2)*Theta(inp, a) + (λL/2)*phi_L**2 + (λR/2)*phi_R**2)
    
    zLt_op = np.einsum('N, NP->P', a_i, xLt)
    zRt_op = np.einsum('N, NP->P', a_i, xRt)
    
    phi_L = Phi(zLt_op)
    phi_R = Phi(zRt_op)  
    
    D = np.sqrt((phi_L-phi_R)**2)
    inp = dF - D**2

    Lt_op = np.mean((Y/2)*λB*D**2 + ((1-Y)/2)*Theta(inp, a) + (λL/2)*phi_L**2 + (λR/2)*phi_R**2)
    
    return L_bays, L_op, Lt_bays, Lt_op

#%%
#import numpy as np
import time

num =1000
a = 1e-2
β = 50
λL=0.
λR=0.
dF=1.
λB=1.
λw = 0.05
λ_damp = 0.1
times = 5

n_iter = 100

al_m = []
al_std = []
vl_m = []
vl_std = []
E_bays_m=[]
E_bays_std=[]
E_op_m=[]
E_op_std=[]
Eg_bays_m=[]
Eg_bays_std=[]
Eg_op_m=[]
Eg_op_std=[]
Ml_m = []
Ql_m = []
Ml_std = []
Ql_std = []
ωL = []
α_l = [1.1,1.2,1.3,1.4]
m = 1
σ = 0.5
for α in α_l:
    since = time.time()

    N = 1000
    P = int(α*N)
    bs = P
    
    x = np.array([1, -1])
    ρB = 0.5
    ρF = 1-ρB
    ρ = np.array([ρB, ρF])
    
    mX = m/N
    σX = σ/np.sqrt(N)

    P_Bpp = int(P*ρB/2)
    P_Bnn = int(P*ρB/2)
    P_Fpn = int(P*ρF/2)
    P_Fnp = int(P*ρF/2)
    
    yLBp = np.ones((P_Bpp,1))
    yRBp = np.ones((P_Bpp,1))
    yLBn = -np.ones((P_Bnn,1))
    yRBn = -np.ones((P_Bnn, 1))
    yLFp = np.ones((P_Fpn,1))
    yRFn = -np.ones((P_Fpn,1))
    yLFn = -np.ones((P_Fnp,1))
    yRFp = np.ones((P_Fnp,1))
    a_l_n = []
    v_l_n = []
    Ml_n = []
    Ql_n = []
    e_bays_n = []
    e_op_n = []
    eg_bays_n = []
    eg_op_n = []
    
    for i in range(times):
        flag = 0
        while flag==0:
            a_iμ_YY, v_iμ_YY = Initial_a_v(N,P_Bpp,P_Bnn,P_Fpn,P_Fnp)
    
            X = []
            for y in [yLBp, yRBp, yLBn, yRBn, yLFp,yRFn,yLFn,yRFp]: 
                Py = y.shape[0]
                x_m = y[:,0]*np.ones((N, Py))*mX
                x_r = np.random.normal(loc=0, scale=σX, size=(N, Py))
                x = x_m + x_r
                X.append(np.expand_dims(x, axis=2))
            
            xLBp = X[0]
            xRBp = X[1]
            xLBn = X[2]
            xRBn = X[3]
            xLFp = X[4]
            xRFn = X[5]
            xLFn = X[6]
            xRFp = X[7]
            #print('xLBp',xLBp.shape)
        
        
        
            XBpp = np.concatenate((xLBp,xRBp), axis=2)
            XBnn = np.concatenate((xLBn,xRBn), axis=2)
            XFpn = np.concatenate((xLFp,xRFn), axis=2)    
            XFnp = np.concatenate((xLFn,xRFp), axis=2)
            X_YY = [XBpp,XBnn,XFpn,XFnp]
        
            YBpp = np.concatenate((yLBp,yRBp), axis=1)
            YBnn = np.concatenate((yLBn,yRBn), axis=1)
            YFpn = np.concatenate((yLFp,yRFn), axis=1)    
            YFnp = np.concatenate((yLFn,yRFp), axis=1)
            YY = [YBpp,YBnn,YFpn,YFnp]
           
            ZLT, ZRT = zt_rand(P_Bpp, P_Bnn, P_Fpn, P_Fnp, N,num)
    
            ΔEωL = []
            ΔCωL = []
            ΔEγL = []
            ΔCγL = []
            ΔEVL = []
            ΔEΓL = []
            ml = []
            vl = []  
            
            m0 = 0
            v0 = 0
            
            for t in range(n_iter):
                
                print('t,',t)
                print('α', α)
                
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
                ω_YY, V_YY = Update_omega_V(a_iμ_YY, v_iμ_YY, X_YY)
                
                ZL_YY,ZR_YY = z_fun(ω_YY,  V_YY, num, ZLT, ZRT)

                Gout = g_out(ω_YY,  V_YY, YY, ZL_YY,ZR_YY, λL, λR, dF,λB, β, a)
                DGout = dg_out(Gout, ω_YY, V_YY,  YY, ZL_YY,ZR_YY, λL, λR, dF,λB, β, a)
                
                γ_μi_YY, Γ_μi_YY, γ_i_YY, Γ_i_YY = Gamma_gamma(X_YY, Gout, DGout)
                
                a_iμ0_YY, v_iμ0_YY = av_iμ(γ_μi_YY, Γ_μi_YY, β,λw)
                a_i, v_i = av(γ_i_YY, Γ_i_YY, β,λw)
                
                if np.any(v_i <= 0):
                    print('V < 0')

                    break

                L_bays, L_op,Lt_bays, Lt_op =  L(a_i,v_i, YY, X_YY, λL, λR,dF, λB,ρF,ρB, mX,σX, N,a, n=int(10000))
            
                m_ =  np.mean(np.sqrt(a_i**2))
                v_ = np.mean(v_i)
                ml.append(m_)
                vl.append(v_)
    
                print('a',np.mean(np.sqrt(a_i*a_i)))
                print('v', np.mean(np.sqrt(v_i*v_i))) 
                print('M',np.mean((a_i)))
                print('Q',np.mean((a_i*a_i+v_i)))
                
                print('L_bays', L_bays)
                print('Lt_bays', Lt_bays)

     
                if np.max(np.abs(np.stack([np.array(a_iμ0_YY)-np.array(a_iμ_YY), np.array(v_iμ0_YY)-np.array(v_iμ_YY)], axis=0))) < λ_damp*1e-3:
                    flag = 1 
                    break
                else:
                    m0 = m_
                    v0 = v_
                    for i in range(len(a_iμ0_YY)):
                        a_iμ_YY[i] = λ_damp*a_iμ_YY[i] + (1-λ_damp)*a_iμ0_YY[i]
                        v_iμ_YY[i] = λ_damp*v_iμ_YY[i] + (1-λ_damp)*v_iμ0_YY[i]
                
                
            import matplotlib.pyplot as plt
            plt.plot(ml, label='m')
            plt.plot(vl, label='v')
            plt.xlabel('epoch\α='+str(α)+' flag: '+str(flag))
            plt.legend()
            plt.show() 
                    

        a_l_n.append(np.mean(a_i))
        v_l_n.append(np.mean(v_i))
        Ml_n.append(np.mean(a_i))
        Ql_n.append(np.mean((a_i*a_i+v_i)))
        e_bays_n.append(L_bays)
        e_op_n.append(L_op)
        eg_bays_n.append(Lt_bays)
        eg_op_n.append(Lt_op)
        
    al_m.append(np.mean(a_l_n))
    al_std.append(np.std(a_l_n))
    vl_m.append(np.mean(v_l_n))
    vl_std.append(np.std(v_l_n))
    Ql_m.append(np.mean(Ql_n))
    Ml_m.append(np.mean(np.abs(Ml_n)))
    Ql_std.append(np.std(Ql_n))
    Ml_std.append(np.std(np.abs(Ml_n)))
    E_bays_m.append(np.mean(e_bays_n))
    E_bays_std.append(np.std(e_bays_n))
    E_op_m.append(np.mean(e_op_n))
    E_op_std.append(np.std(e_op_n))
    Eg_bays_m.append(np.mean(eg_bays_n))
    Eg_bays_std.append(np.std(eg_bays_n))
    Eg_op_m.append(np.mean(eg_op_n))
    Eg_op_std.append(np.std(eg_op_n)) 
    