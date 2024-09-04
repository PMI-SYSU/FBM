"""
Created on Fri Jun 16 17:41:39 2023

@author: 31458
"""

import numpy as np

def Initial_MQq():
    M = 1 
    Q = 1
    q = np.random.rand()*Q
    
    return M,Q,q

def Phi(m, σ, M, Q, q,ta,t):
    
    inpu_= m*M + σ*(np.sqrt(Q-q)*ta+np.sqrt(q)*t)
    phi = np.tanh(inpu_)
    return phi

def DPhi(m, σ, M, Q,q,ta,t):
    
    inpu_= m*M + σ*(np.sqrt(Q-q)*ta+np.sqrt(q)*t)
    dphi = 1 - np.tanh(inpu_)*np.tanh(inpu_)
    
    return dphi

def DDPhi(m, σ, M, Q,q,ta,t):
    
    inpu_= m*M + σ*(np.sqrt(Q-q)*ta+np.sqrt(q)*t)
    ddphi = -2*np.tanh(inpu_)*(1 - np.tanh(inpu_)*np.tanh(inpu_))
    
    return ddphi

def Update_Q(Q_, q_, M_, λw):
    
    Q = (M_**2 + q_)/(Q_ + q_ + λw)**2 + 1/(Q_ + q_ +λw)
    
    return Q

def Update_q(Q_, q_, M_, λw):
    
    q = (M_**2 + q_)/(Q_ + q_ +λw)**2 
    
    return q

def Update_M(Q_, q_, M_, λw):
    
    M = M_/(Q_ + q_ +λw)
    
    return M

def Randomness(sample_num):
    tL = np.random.normal(loc=0.0, scale=1.0, size=(sample_num,1))
    tR = np.random.normal(loc=0.0, scale=1.0, size=(sample_num,1)) 
    
    taL = np.random.normal(loc=0.0, scale=1.0, size=(1,sample_num)) 
    taR = np.random.normal(loc=0.0, scale=1.0, size=(1,sample_num)) 
    return tL,tR, taL,taR    

def Theta(x, a):
    y = (x + np.sqrt(x**2 + a))/2
        
    return y

def dTheta(x, a):
    dy = (1 + x/np.sqrt(x**2 + a))/2

    return dy 

def ddTheta(x, a):
    ddy = a/(2*np.sqrt(x**2 + a)*(x**2 + a))
        
    return ddy



def DH(yL, yR, phi_L, phi_R, dphi_L, dphi_R, ddphi_L, ddphi_R,  λB, dF, a):

    Y = np.sign(yL*yR + 1)
    D = np.sqrt((phi_L-phi_R)**2)
    inp = dF - D**2
    
    D2L = 2*(phi_L-phi_R)*dphi_L
    D2LL = 2*(dphi_L**2 + (phi_L-phi_R)*ddphi_L)

    D2R = 2*(phi_L-phi_R)*(-dphi_R)
    D2RR = 2*(dphi_R**2 + (phi_L-phi_R)*(-ddphi_R))
    
    
    H = (Y/2)*λB*D**2 + ((1-Y)/2)*Theta(inp, a)
    
    HL = Y*λB*D2L/2 + (1-Y)*dTheta(inp, a)*(-D2L)/2 
    HR = Y*λB*D2R/2 + (1-Y)*dTheta(inp, a)*(-D2R)/2  
    HLL = Y*λB*D2LL/2 \
            + (1-Y)*(ddTheta(inp, a)*D2L**2/2 + dTheta(inp, a)*(-D2LL/2))
                
            
    HRR =  Y*λB*D2RR/2 \
        + (1-Y)*(ddTheta(inp, a)*D2R**2/2 + dTheta(inp, a)*(-D2RR/2))
     
    return H, HL, HR, HLL, HRR 

def Update_MQq_y1y2(m, σ, Q, q, M, α, β, tL,tR, taL, taR,  λB, dF,ρB,ρF, a):
    dQ_ = 0
    dq_ = 0
    dM_ = 0
    for yL in [1, -1]:
        for yR in [1, -1]:
            if yL ==yR:
                ρ=ρB
            else:
                ρ=ρF
            phi_L = Phi(yL*m, σ, M, Q, q,taL,tL)
            phi_R = Phi(yR*m, σ, M, Q, q,taR,tR) 
         
            dphi_L = DPhi(yL*m, σ, M, Q, q,taL,tL)
            dphi_R = DPhi(yR*m, σ, M, Q, q,taR,tR) 
            
            ddphi_L = DDPhi(yL*m, σ, M, Q, q,taL,tL)
            ddphi_R = DDPhi(yR*m, σ, M, Q, q,taR,tR) 
        
            H, H_L, H_R, H_LL, H_RR  = DH(yL, yR, phi_L, phi_R, dphi_L, dphi_R, ddphi_L, ddphi_R, λB, dF, a)

            dQ0_ = 2*α*β*σ**2*np.sum(np.sum((1/(2*σ*np.sqrt(Q-q))*(H_L*taL+H_R*taR))*np.exp(-β*H), axis=1)/np.sum(np.exp(-β*H), axis=1))/H.shape[0]
           
            dq0_ = α*(β**2*σ**2)*(np.sum((np.sum((H_L)*np.exp(-β*H), axis=1)/np.sum(np.exp(-β*H), axis=1))**2)/H.shape[0])\
                         + α*(β**2*σ**2)*(np.sum((np.sum((H_R)*np.exp(-β*H), axis=1)/np.sum(np.exp(-β*H), axis=1))**2)/H.shape[0])
            dM0_ = -α*β*m*np.sum(np.sum((yL*H_L+yR*H_R)*np.exp(-β*H), axis=1)/np.sum(np.exp(-β*H), axis=1))/H.shape[0]
            
            dQ_ += (ρ/2)*dQ0_
            dq_ += (ρ/2)*dq0_
            dM_ += (ρ/2)*dM0_

    Q_ = dQ_
    q_ = dq_
    M_ = dM_
    
    return Q_, q_, M_

def Loss_saddle_point(M,q, Q, m, σ,λB,dF,λw, ρB,ρF, tL,tR, taL, taR, a, α, β, sample_num=100000):
    H = 0
    for yL in [1,-1]:
        for yR in [1, -1]:
            Y = np.sign(yL*yR + 1)
            
            phi_L = Phi(yL*m, σ, M, Q, q,taL,tL)
            phi_R = Phi(yR*m, σ, M, Q, q,taR,tR)             
            D = np.sqrt((phi_L-phi_R)**2)
            inp = dF - D**2

            H_y = (Y/2)*λB*D**2 + ((1-Y)/2)*Theta(inp, a)
            if Y ==0:            
                Hi = (ρB/2)*np.mean(np.sum(H_y*np.exp(-β*H_y), axis=1)/np.sum(np.exp(-β*H_y), axis=1))
            else:
                Hi = (ρF/2)*np.mean(np.sum(H_y*np.exp(-β*H_y), axis=1)/np.sum(np.exp(-β*H_y), axis=1))
            H += Hi
           
            
    f = -α*H
    εt = -f/α

    εg = 0
    Dg = 0
    for yL in [1,-1]:
        for yR in [1, -1]:
            Y = np.sign(yL*yR + 1)

            zL = np.random.normal(loc=yL*m*M, scale=σ*np.sqrt(Q), size=(sample_num,1))
            zR = np.random.normal(loc=yR*m*M, scale=σ*np.sqrt(Q), size=(sample_num,1))
            
            phi_L= np.tanh(zL)
            phi_R= np.tanh(zR)
            
            D = np.sqrt((phi_L-phi_R)**2)
            inp = dF - D**2
            
            if Y ==0:            
                L = np.sum((1/4)*((Y/2)*λB*D**2 + ((1-Y)/2)*Theta(inp, a) ))/(sample_num) 
                Dg += np.mean(D**2)/2

            else:
                L = np.sum((1/4)*((Y/2)*λB*D**2 + ((1-Y)/2)*Theta(inp, a) ))/(sample_num) 
            
            εg += L
    
    return εt, εg, Dg 

def Update_MQq_RS(rou, m, σ, Q, q, M, α, β, t1,t2, ta1,ta2, λB, dF, ρB,ρF, a,sample_num):

    dQ_, dq_, dM_ = Update_MQq_y1y2(m, σ, Q, q, M, α, β, t1,t2, ta1,ta2, λB, dF, ρB,ρF,a)

    return dQ_, dq_, dM_

#%%RS迭代    
import time

ρB = 0.5 
β = 50
λw = β*0.05


dF_L = [4.2] 
λB = 1

m = 1
rou= 0.5
rou_= 1
i=0
Ql = []
ql = []
Ml = []
FF= []
FB =[]
Lt = []
Lg = []
DL = []
Q_es = 1
q_es = 1
M_es = 1
sample_num= 1000
a = 1e-2
es=1e-3*rou

Δs = np.arange(0.1,2.1,0.1) 
αs =  [2.5]
αs = np.arange(0.1,3,0.1)
α = 2.5 
Δ = 0.5 
 
for dF in dF_L:
    ρF = 1 - ρB
    since = time.time()

    Q = 1
    q = np.random.rand()*Q
    M = np.random.rand()
    Q_ = 0
    q_ = 0
    M_ = 0
    Q_list = []
    q_list = []
    M_list = []
    Ka_list = []
    L_list = []

    tL,tR, taL, taR = Randomness(sample_num)
    Q_es = 1
    q_es = 1
    M_es = 1
    e=0
    while  (Q_es>=es)or(q_es>=es)or(M_es>=es):
        e = e+1
        
        print('epoch:',e)
        print('α:',α)
        print('Δ:',Δ)
        print('dF:',dF)
        print('ρB:',ρB)
 
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        dQ_, dq_, dM_ = Update_MQq_RS(rou, m, Δ, Q, q, M, α, β,  tL,tR, taL, taR, λB, dF,ρB,ρF, a, sample_num)        
        
        print('Q_', Q_)
        print('q_', q_)    
        print('M_', M_)    
    
        dQ = Update_Q(Q_, q_, M_, λw)
        dq = Update_q(Q_, q_, M_, λw)
        dM = Update_M(Q_, q_, M_, λw)
        
        Q_es=np.abs(Q-((1-rou)*Q + rou*dQ))
        q_es=np.abs(q-((1-rou)*q + rou*dq))
        M_es=np.abs(M-((1-rou)*M + rou*dM))
        
        Q = (1-rou)*Q + rou*dQ
        q = (1-rou)*q + rou*dq
        M = (1-rou)*M + rou*dM
        Ka = β*(Q-q)    
        
        if q>=Q:
            print('q>Q')
        
        Q_list.append(Q)
        q_list.append(q)
        M_list.append(M)
        Ka_list.append(Ka)
        print('Q', Q)
        print('q', q)    
        print('M', M)  
 
        εt, εg, Dg  = Loss_saddle_point(M, q, Q, m, Δ,λB,dF,λw, ρB,ρF, tL,tR, taL, taR, a, α, β)     

        print('εt',εt)
        print('εg', εg)
        print('Dg', Dg)

        
        if q==0:
            break
    Ql.append(Q)
    ql.append(q)
    Ml.append(M)
    Lt.append(εt)
    Lg.append(εg)
    DL.append(Dg)


#%%准确率函数

from scipy import special

def Acc_fun(m, Δ, Ml, Ql):
    
    Acc = (1/4)*(special.erf(m*Ml/(2*Δ*np.sqrt(Ql))) - special.erf(-m*Ml/(2*Δ*np.sqrt(Ql)))) + 1/2

    return Acc


#%%对抗性误差与准确率

def Theta_fun(z):
    tmp = z.copy()
    tmp[tmp < 0] = 0
    tmp[tmp > 0] = 1
    return tmp

def phi(z):
    return np.tanh(z)

def dphi(z):
    return 1 - (np.tanh(z))**2

def L2attack(M, q, m, Δ, ε, num):
    E_adv = 0
    for y in [+1, -1]:
        z = np.random.normal(loc=y*m*M, scale=Δ*np.sqrt(q), size=(num,1))
        
        K = -(y-phi(z))*dphi(z)
        sgnK = np.sign(K)
        
        E_adv += np.mean((1/2)*(y - np.tanh(z + ε*sgnK*np.sqrt(q)))**2)/2
        
    z = np.random.normal(loc=m*M, scale=Δ*np.sqrt(q), size=(num,1))
    K = -(1-phi(z))*dphi(z)
    sgnK = np.sign(K)   
    Acc_adv0 =(np.mean(Theta_fun((z + ε*np.sqrt(q)*sgnK))))
   
    return E_adv, Acc_adv0



ε_L = np.arange(0.1,1.1,0.1)
m=1
λF_L = np.arange(0.1,5.1,0.1)
num = 10000
Sl0 = []
Sl1 = []

for i in range(len(λF_L)):
    M = Ml[i]
    q = Ql[i]
    S0_adv = 0
    S1_adv = 0
    Acc = []
    for ε in ε_L:
        E_adv, Acc_adv, Acc_adv0  = L2attack(M, q, m, Δ, ε, num)
        Acc.append(Acc_adv)
        S0_adv += E_adv*0.1
        S1_adv += Acc_adv*0.1
    
    Sl0.append(S0_adv)
    Sl1.append(S1_adv)




