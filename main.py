#二维寻优 灰狼算法
import numpy as np
import numpy.random as rd
from math import sqrt, cos

#20是SearchAgens_no,2是dim
#lb是下界，ub是上界，dim是寻址变量个数，SearchAgents_no是寻值狼的数量，Max_iter最大迭代次数

#原始灰狼算法
iterations=[]
fina_fitness=[]

def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter):
    #初始化三头狼
    Alpha_pos=np.zeros(dim)
    Alpha_score=float('inf')
    
    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=np.zeros(dim)
    Delta_score=float('inf')
    
    Positions=np.zeros((SearchAgents_no,dim))
    Positions = np.dot(rd.rand(SearchAgents_no,dim),(ub-lb))+lb #初始化首次搜索位置
    Convergence_curve=np.zeros((1,Max_iter))#初始化融合曲线
    
    index_iter=0
    while index_iter<Max_iter:
        
        #遍历
        for i in range(0,(Positions.shape[0])):
            for j in range(0,(Positions.shape[1])):
                flag4ub=Positions[i,j]>ub
                flag4lb=Positions[i,j]<lb
                if flag4ub:
                    Positions[i,j]=ub
                if flag4lb:
                    Positions[i,j]=lb
            fitness=objf(Positions[i,0],Positions[i,1])
        
        if fitness<Alpha_score:
            Alpha_score=fitness
            Alpha_pos=Positions[i]
        if fitness>Alpha_score and fitness<Beta_score:
            Beta_score=fitness
            Beta_pos=Positions[i]
        if fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score:
            Delta_score=fitness
            Delta_pos=Positions[i]
            
            
        a=2-index_iter*(2/Max_iter)
    
    #遍历狼
        for i in range(0,(Positions.shape[0])):  #（0,20）
            for j in range(0,(Positions.shape[1])):  #（0,2）
                r1=rd.random(1)
                r2=rd.random(1)               
                A1=2*a*r1-a # 计算系数A
                C1=2*r2 # 计算系数C

            #Alpha狼位置更新
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j])
                X1=Alpha_pos[j]-A1*D_alpha
                    
                r1=rd.random(1)
                r2=rd.random(1)

                A2=2*a*r1-a
                C2=2*r2

            # Beta狼位置更新
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j])
                X2=Beta_pos[j]-A2*D_beta

                r1=rd.random(1)
                r2=rd.random(1)

                A3=2*a*r1-a
                C3=2*r2

                # Delta狼位置更新
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j])
                X3=Delta_pos[j]-A3*D_delta

            # 位置更新
                Positions[i,j]=(X1+X2+X3)/3
            
        index_iter = index_iter + 1
        iterations.append(index_iter)
        fina_fitness.append(fitness)
    
        print('迭代次数'+str(index_iter)+'最优值是'+str(fitness))
        
    
    return fina_fitness,iterations

#Ackle函数
def F1(x1,x2):
    s=(-20)*np.exp((-0.2)*np.sqrt((x1**2+x2**2)/2))-np.exp((np.cos(2*np.pi*x1+2*np.pi*x2)/2))+20+np.exp(1) #[-32,32]
    return s



#Griewank函数
def F2(x1,x2):
    s=((x1**2+x2**2)/4000)-np.dot(cos(x1),cos(x2/sqrt(2)))+1   #[-600,600]
    return s


#主程序
Max_iter = 30#迭代次数
lb = -32#下界
ub = 32#上界
dim = 2#狼的寻值范围
SearchAgents_no = 20#寻值的狼的数量
x = GWO(F1, lb, ub, dim, SearchAgents_no, Max_iter) #这个x随便取

import matplotlib.pyplot as plt
plt.figure()
fig=plt.gcf()
fig.set_size_inches(18,10)
plt.xlabel("iterations",size=20)
plt.ylabel("fina_fitness",size=20)
plt.plot(fina_fitness,color="r",linewidth=5)
plt.show()
