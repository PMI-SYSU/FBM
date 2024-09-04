# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:57:08 2024

@author: 31458
"""

import time
import numpy as np

#%% Cavity: εg-α 
from FBM_Cavity import Initial_a_v, zt_rand, Update_omega_V, z_fun, g_out, dg_out, Gamma_gamma, av_iμ,av, L

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
Eg_bays_m=[]
Eg_bays_std=[]

m = 1
Δ = 0.5
α_l = [1.1,1.2,1.3,1.4]

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
    ΔX = Δ/np.sqrt(N)

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
                x_r = np.random.normal(loc=0, scale=ΔX, size=(N, Py))
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

                L_bays, L_op,Lt_bays, Lt_op =  L(a_i,v_i, YY, X_YY, λL, λR,dF, λB,ρF,ρB, mX,ΔX, N,a, n=int(10000))
            
                m_ =  np.mean(np.sqrt(a_i**2))
                v_ = np.mean(v_i)
                ml.append(m_)
                vl.append(v_)
    
                print('a',np.mean(np.sqrt(a_i*a_i)))
                print('v', np.mean(np.sqrt(v_i*v_i))) 

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

        e_bays_n.append(L_bays)
        eg_bays_n.append(Lt_bays)
        
    al_m.append(np.mean(a_l_n))
    al_std.append(np.std(a_l_n))
    vl_m.append(np.mean(v_l_n))
    vl_std.append(np.std(v_l_n))

    E_bays_m.append(np.mean(e_bays_n))
    E_bays_std.append(np.std(e_bays_n))

    Eg_bays_m.append(np.mean(eg_bays_n))
    Eg_bays_std.append(np.std(eg_bays_n))

#%%  Cavity: εg-Δ

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
Eg_bays_m=[]
Eg_bays_std=[]

m = 1
Δ = np.arange(0.1,1.7,0.1)
α = 2.5

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
    ΔX = Δ/np.sqrt(N)

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
                x_r = np.random.normal(loc=0, scale=ΔX, size=(N, Py))
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

                L_bays, L_op,Lt_bays, Lt_op =  L(a_i,v_i, YY, X_YY, λL, λR,dF, λB,ρF,ρB, mX,ΔX, N,a, n=int(10000))
            
                m_ =  np.mean(np.sqrt(a_i**2))
                v_ = np.mean(v_i)
                ml.append(m_)
                vl.append(v_)
    
                print('a',np.mean(np.sqrt(a_i*a_i)))
                print('v', np.mean(np.sqrt(v_i*v_i))) 

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

        e_bays_n.append(L_bays)
        eg_bays_n.append(Lt_bays)
        
    al_m.append(np.mean(a_l_n))
    al_std.append(np.std(a_l_n))
    vl_m.append(np.mean(v_l_n))
    vl_std.append(np.std(v_l_n))

    E_bays_m.append(np.mean(e_bays_n))
    E_bays_std.append(np.std(e_bays_n))

    Eg_bays_m.append(np.mean(eg_bays_n))
    Eg_bays_std.append(np.std(eg_bays_n))
    

#%%RS    
import time
from FBM_RS import Randomness,Update_MQq_RS, Update_Q, Update_q, Update_M, Loss_saddle_point

ρB = 0.5 
β = 50
λw = β*0.05


dF_L = 1
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
αs = np.arange(0.1,3,0.1)
α = 2.5 
Δ = 0.5 
 
#%% RS εg-α 
for α in αs:
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
#%% RS εg-Δ
α = 2.5 

for Δ in Δs:
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


#%% RS εg-dF
α = 2.5 
Δ = 0.5
dF_L = np.arange(0.1,5.1,0.1)

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
    
from FBM_RS import Acc_fun
Acc_dF = Acc_fun(m, Δ, Ml, Ql)



#%% L2对抗性曲线

from FBM_RS import L2attack

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
    

