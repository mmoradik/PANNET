import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from graphviz import Digraph
import pydotplus
import timeit
from tqdm import tqdm

#---------------------------------------------------------------------------------------------

def initialization():
    global PopSize,NumGen,max_neurons,min_neurons,max_context,obsN,inN,max_input,max_output,con_rate
    global crossover_rate,mutation_rate,elit_rate,tournrate,B_max_w,w_rate
    PopSize=100
    NumGen=2000
    max_neurons=8      
    min_neurons=4
    max_context=7
    obsN=1          
    inN=0              
    max_input=max_context+obsN+inN     
    max_output=max_context+obsN        
    con_rate=.5                      

    B_max_w=.5
    w_rate=0.1
    mutation_rate =0.01 
    elit_rate=0.1
    crossover_rate=0.5
    tournrate=0.1
    
    
#####---------------------------------------------------------------------------------------------------------
def Evolution(Data,y_a,n_training,Sc):
    
    initialization()
   
    X0=0.01*np.ones(max_input)
    n_fold=5
    lenfold=int(np.floor(n_training/5)) 
    
    TrainingData=Data[:n_training]
    Train_orig=y_a[:n_training]  
    Test_orig=y_a[n_training:]
    
    MinFit=np.empty(NumGen)
    MaxFit=np.empty(NumGen)
    AvgFit=np.empty(NumGen)
    UNI=np.empty(NumGen)
    Valerror=np.empty(NumGen)
    test_MSE=np.empty(NumGen)


    Predicted_output=[]
    Network=[]
    
    Fitness=np.zeros((n_fold,PopSize))
    Pop=[GenerateIndividual() for i in range(0,PopSize)]
    
    for i in range(0,PopSize):
        Fitness[:,i]=CalcFitness(Pop[i],TrainingData,X0,n_fold,lenfold)  
  

    ######-----------------
    Wmut=int(NumGen)/2
    elit_n=int(np.floor(elit_rate*PopSize))
    newsize=PopSize+elit_n
    cross=int(PopSize/2)
    
    for j in tqdm(range(0,NumGen)):      
        ind_best=Fitness[n_fold-1].argsort()[:elit_n].copy()      
        elit_ind=[Pop[int(a)].copy() for a in ind_best].copy()
 
        selectedpop=[]
        for i in range(0,PopSize):
            tournindex=tournamentSelection(Fitness[n_fold-1],PopSize,tournrate)
            selectedpop.append(Pop[tournindex].copy())
            
            
        crosspop=[]
        k=range(0,PopSize)
        k=random.sample(k,PopSize)

        for i in range(0,cross):  
            if random.random() < crossover_rate: 
                CrossChild1,CrossChild2= Crossover(selectedpop[k[2*i]].copy(),selectedpop[k[2*i+1]].copy(),crossover_rate)
            else:
                CrossChild1=selectedpop[k[2*i]].copy()
                CrossChild2=selectedpop[k[2*i+1]].copy()
            crosspop.append(CrossChild1)
            crosspop.append(CrossChild2)

    ########-----------
        newpop=[]
        if j <=Wmut:
            for i in range(0,PopSize):
                newpop.append(Mutation(crosspop[i].copy(),mutation_rate,w_rate))
        else:
            for i in range(0,PopSize): 
                newpop.append(WMutation(crosspop[i].copy(),mutation_rate,w_rate))
        
        newpop.extend(elit_ind)
     
        #######-----------
        
        newFitness=np.zeros((n_fold,newsize))
        for i in range(0,newsize):
            newFitness[:,i]=CalcFitness(newpop[i],TrainingData,X0,n_fold,lenfold) 
        ###---------
            
        index_fittest=newFitness[n_fold-1].argsort()[:PopSize].copy()
        Pop=[newpop[int(a)].copy() for a in index_fittest] .copy()
        Fitness=newFitness[:,index_fittest].copy()
        
        index_fittest_fold=Fitness[:n_fold-1].argsort()[:,:1].copy()
        Pop_fold=[[Pop[int(a)].copy() for a in index_fittest_fold[fold]] for fold in range(n_fold-1)].copy()  
        
        ###----------

        valid_error=np.zeros(n_fold-1)
        for fold in range(n_fold-2):
            x_p,network_=Valid(Pop_fold[fold],1,X0,TrainingData[:lenfold*(fold+2)]) 
            x_hat=x_p[lenfold*(fold+1):lenfold*(fold+2)]
            valid_error[fold]=MSE(TrainingData[lenfold*(fold+1):lenfold*(fold+2)],x_hat)  
            
        #the last fold may has more samples than others
        x_p,network_=Valid(Pop_fold[n_fold-2],1,X0,TrainingData) 
        x_hat=x_p[lenfold*(n_fold-1):]
        valid_error[n_fold-2]=MSE(TrainingData[lenfold*(n_fold-1):],x_hat)

        Valerror[j]=np.mean(valid_error)
        ######----------

        y_p,network=Valid(Pop,Fitness[n_fold-1],X0,Data)
        Network.append(network)

        #############------------------------------------

        test_MSE[j]=MSE(Data[n_training:],y_p[n_training:])
        ########-------------
        MinFit[j]=min(Fitness[n_fold-1,:])
        MaxFit[j]=max(Fitness[n_fold-1])
        AvgFit[j]=np.mean(Fitness[n_fold-1])
        UNI[j]=len(np.unique(Fitness[n_fold-1]))
 
    return MinFit,MaxFit,AvgFit,UNI,Valerror,test_MSE,Network,X0
  
    
###-------------------------------    
def GenerateIndividual():
    x=False
    while x==False:
        NumNeurons=random.randint(min_neurons,max_neurons)
        Individual=np.array([NumNeurons])
        check=[]
        for i in range(NumNeurons):
            NumInputs=random.randint(1,np.floor(max_input*con_rate))
            NumOutputs=random.randint(1,np.floor(max_output*con_rate))
            InputIndices=random.sample(range(0, max_input), NumInputs)
            OutputIndices=random.sample(range(inN, max_input), NumOutputs)  
            check.extend(OutputIndices)
            WeightInput=B_max_w * np.random.uniform(-1, 1,NumInputs)
            WeightOutput=B_max_w * np.random.uniform(-1, 1,NumOutputs)
            Individual=np.hstack( [Individual,NumInputs,NumOutputs,InputIndices,OutputIndices,WeightInput,WeightOutput])
        if 0 in check:
            x=True
    
    return Individual


    #----------------------------------------------------------------------------------------------------
         
def CalcFitness(Individual,TrainingData,X,n_fold,lenfold):
    T=len(TrainingData)   
    NumNeuron=int(Individual[0])
    OutputConnected=[]

    A=np.zeros([NumNeuron,max_input])
    B=np.zeros([max_output,NumNeuron])

    k=1
    s_w=0
    K=0
    for i in range(0,NumNeuron):
        NumInputs=int(Individual[k])
        NumOutputs=int(Individual[k+1])
        
        Inputnodes=[int(x) for x in Individual[k+2:k+2+NumInputs] ]
        Outputnodes=[int(x) for x in Individual[k+2+NumInputs:k+2+NumInputs+NumOutputs ] ]
        
        OutputConnected.extend(Outputnodes)
        
        InputWeights=Individual[k+2+NumInputs+NumOutputs :k+2+ 2 *NumInputs+NumOutputs  ]
        OutputWeights=Individual[k+2 + 2*NumInputs+NumOutputs :k+2 +2 *NumInputs+2* NumOutputs  ]
        s_w=s_w+sum(abs(InputWeights))+sum(abs(OutputWeights))
        K=K+NumInputs+NumOutputs 
        k=k+2 +2 *NumInputs+2* NumOutputs
        
        A[i,Inputnodes]=InputWeights
        B[Outputnodes,i]=OutputWeights

    UnconnectedOutputs=list(set(range(0,max_input))-set(OutputConnected))    
    
    Fitness=100*np.ones(n_fold)
    if 0 not in UnconnectedOutputs:
       
        XNext=X.copy()
        XNext[0]=TrainingData[0].copy()
        
        Fitness=np.zeros(n_fold)
        fold=0
        s=0
        for t in range(1,T):
            XNext=np.dot(B,np.tanh(np.dot(A,XNext)))
            XNext[UnconnectedOutputs]=X[UnconnectedOutputs].copy()
            x_hat=XNext[0].copy()
            
            if t==max_context:
                s=0

            s=s +  (x_hat-TrainingData[t][0])**2      
            XNext[0]=TrainingData[t].copy()

            if fold==n_fold-1 and t==T-1:
                Fitness[fold]=(s/(T-max_context)).copy()
                #fold=fold+1 
            elif fold!=n_fold-1 and t==(lenfold*(fold+1)-1):
                Fitness[fold]=(s/(lenfold*(fold+1)-max_context)).copy()
                fold=fold+1
                        
    return Fitness   
    ##---------------------------------------------------------------------------------------------------------

def Crossover(Parent1,Parent2,crossover_rate):
    flag=False
    while flag==False:
        
        IndexNeuron1=random.randint(0,Parent1[0]-1)
        IndexNeuron2=random.randint(0,Parent2[0]-1)


        Partitions1=[]
        Partitions2=[]

        k=1
        for i in range(0,int(Parent1[0])):
            NumInputs=int(Parent1[k])
            NumOutputs=int(Parent1[k+1])
            Partitions1.append(Parent1[k:k+2 +2 *NumInputs+2* NumOutputs])
            k=k+2 +2 *NumInputs+2* NumOutputs

        k=1
        for i in range(0,int(Parent2[0])):
            NumInputs=int(Parent2[k])
            NumOutputs=int(Parent2[k+1])
            Partitions2.append(Parent2[k:k+2 +2 *NumInputs+2* NumOutputs])
            k=k+2 +2 *NumInputs+2* NumOutputs 

        x=Partitions1[IndexNeuron1].copy()
        y=Partitions2[IndexNeuron2].copy()

        Partitions1[IndexNeuron1]=y.copy()
        Partitions2[IndexNeuron2]=x.copy()

        Child1=np.array([Parent1[0]])
        for i in range(0,int(Parent1[0])):
            Child1=np.hstack([Child1,Partitions1[i]])

        Child2=np.array([Parent2[0]])
        for i in range(0,int(Parent2[0])):
            Child2=np.hstack([Child2,Partitions2[i]])
            
        ####--------
        OutputConnected1,OutputConnected2=[],[]
        k=1
        for i in range(0,int(Child1[0])):
            NumInputs=int(Child1[k])
            NumOutputs=int(Child1[k+1])
            Outputnodes=[int(x) for x in Child1[k+2+NumInputs:k+2+NumInputs+NumOutputs ] ]
            OutputConnected1.extend(Outputnodes)
            k=k+2 +2 *NumInputs+2* NumOutputs
        k=1
        for i in range(0,int(Child2[0])):
            NumInputs=int(Child2[k])
            NumOutputs=int(Child2[k+1])
            Outputnodes=[int(x) for x in Child2[k+2+NumInputs:k+2+NumInputs+NumOutputs ] ]
            OutputConnected2.extend(Outputnodes)
            k=k+2 +2 *NumInputs+2* NumOutputs 
            
        if 0 in OutputConnected1 and 0 in OutputConnected2:
            flag=True
            
    return Child1,Child2


#----------------------------------------------------------------------------------------
def WMutation(Parent,mutation_rate,w_rate):
    NumNeurons=int(Parent[0])
    Parent_new=Parent.copy()

    k_old=1
    for neuron in range(0,NumNeurons): #
        NumInputs=int(Parent[k_old])
        NumOutputs=int(Parent[k_old+1])

        InputWeights=Parent[k_old+2+NumInputs+NumOutputs :k_old+2+ 2 *NumInputs+NumOutputs  ]
        OutputWeights=Parent[k_old+2 + 2*NumInputs+NumOutputs :k_old+2 +2 *NumInputs+2* NumOutputs  ]
        
        for link in range(0,NumInputs):
            if random.random() < mutation_rate:       #mutate input wights
                InputWeights[link]=np.random.normal(InputWeights[link],w_rate*B_max_w*2)
        for link in range(0,NumOutputs):
            if random.random() < mutation_rate:       #mutate output wights
                OutputWeights[link]=np.random.normal(OutputWeights[link],w_rate*B_max_w*2)   

                              
        Parent_new[k_old+2+NumInputs+NumOutputs :k_old+2+ 2 *NumInputs+NumOutputs  ]=  InputWeights 
        Parent_new[k_old+2 + 2*NumInputs+NumOutputs :k_old+2 +2 *NumInputs+2* NumOutputs  ]=OutputWeights
        k_old=k_old+2 +2 *NumInputs+2* NumOutputs 
    return Parent_new           
        
#----------------------------        
def Mutation(Parent_old,mutation_rate,w_rate):
    flag=False
    while flag==False:
        Parent=Parent_old.copy()
        NumNeurons=int(Parent[0])

        ###########
        if NumNeurons==min_neurons:
            x=0
        elif NumNeurons==max_neurons:
            x=1
        else:
            if random.randint(0,1)==0:
                x=0
            else:
                x=1
        #####################
        if random.random() < mutation_rate and x==0: ##add one neuron
            NumNeurons=NumNeurons+1
            NumInputs=random.randint(1,np.floor(max_input*con_rate))
            NumOutputs=random.randint(1,np.floor(max_output*con_rate))
            InputIndices=random.sample(range(0, max_input), NumInputs)
            OutputIndices=random.sample(range(inN, max_input), NumOutputs)     
            WeightInput=B_max_w * np.random.uniform(-1, 1,NumInputs)
            WeightOutput=B_max_w * np.random.uniform(-1, 1,NumOutputs)
            Parent=np.hstack( [Parent,NumInputs,NumOutputs,InputIndices,OutputIndices,WeightInput,WeightOutput])
            Parent[0]=NumNeurons

        elif random.random() < mutation_rate and x==1: ##delete one neuron
            delneuron=random.randint(0,NumNeurons-1)
            Partitions=[]
            k=1
            for i in range(0,NumNeurons):
                NumInputs=int(Parent[k])
                NumOutputs=int(Parent[k+1])
                Partitions.append(Parent[k:k+2 +2 *NumInputs+2* NumOutputs])
                k=k+2 +2 *NumInputs+2* NumOutputs
            del Partitions[delneuron]    
            NumNeurons=NumNeurons-1 
            Parent=np.array([NumNeurons])   
            for i in range(0,NumNeurons):
                Parent=np.hstack([Parent,Partitions[i]])
        
        ######-------------------
        NumNeurons=int(Parent[0])
        Parent_new=np.array([NumNeurons])

        k_old=1
        #mutation on input connections
        for neuron in range(0,NumNeurons): 
            NumInputs=int(Parent[k_old])
            NumOutputs=int(Parent[k_old+1])


            Inputnodes=[int(x) for x in Parent[k_old+2:k_old+2+NumInputs] ]
            Outputnodes=[int(x) for x in Parent[k_old+2+NumInputs:k_old+2+NumInputs+NumOutputs ] ]

            InputWeights=Parent[k_old+2+NumInputs+NumOutputs :k_old+2+ 2 *NumInputs+NumOutputs  ]
            OutputWeights=Parent[k_old+2 + 2*NumInputs+NumOutputs :k_old+2 +2 *NumInputs+2* NumOutputs  ]

            #-----------------------------------------------

            for link in range(0,NumInputs):

                if random.random() < mutation_rate:     #mutate input nodes
                    remain_A=[]
                    A=list(range(max_input))
                    remain_A=list(set(A)-set(Inputnodes))   #candidates for rewiring

                    if len(remain_A)>0:
                        index=random.randint(0,len(remain_A)-1)
                        node=remain_A[index]
                        Inputnodes[link]=node


                if random.random() < mutation_rate:       #mutate input wights
                    InputWeights[link]=np.random.normal(InputWeights[link],w_rate*B_max_w*2)

            #-----------------------------------------------

            for link in range(0,NumOutputs):

                if random.random() < mutation_rate:        #mutate output nodes
                    remain_A=[]
                    A=list(range(max_input))

                    # consider external inputs, later
                    remain_A=list(set(A)-set(Outputnodes))   #candidates for rewiring

                    if len(remain_A)>0:
                        index=random.randint(0,len(remain_A)-1)
                        node=remain_A[index]
                        Outputnodes[link]=node

                if random.random() < mutation_rate:       #mutate output wights
                    OutputWeights[link]=np.random.normal(OutputWeights[link],w_rate*B_max_w*2)   

            k_old=k_old+2 +2 *NumInputs+2* NumOutputs


            ########### mutation on number of incomming connections             
            if random.random() < mutation_rate:
                if NumInputs==1:
                    NumInputs=NumInputs+1
                    InputWeights=np.append(InputWeights,B_max_w * np.random.uniform(-1, 1,1))

                    remain_A=[]
                    A=list(range(max_input))
                    remain_A=list(set(A)-set(Inputnodes))
                    index=random.randint(0,len(remain_A)-1)
                    node=remain_A[index]
                    Inputnodes=np.append(Inputnodes,node)


                elif NumInputs==np.floor(max_output*con_rate): #max_input   
                    NumInputs=NumInputs-1
                    link=random.randint(0,NumInputs)
                    Inputnodes=np.delete(Inputnodes,link) 
                    InputWeights=np.delete(InputWeights,link) 

                else:
                    if random.randint(0,1)==0:
                        NumInputs=NumInputs+1
                        InputWeights=np.append(InputWeights,B_max_w * np.random.uniform(-1, 1,1))

                        remain_A=[]
                        A=list(range(max_input))
                        remain_A=list(set(A)-set(Inputnodes))
                        index=random.randint(0,len(remain_A)-1)
                        node=remain_A[index]
                        Inputnodes=np.append(Inputnodes,node)


                    else:    
                        NumInputs=NumInputs-1
                        link=random.randint(0,NumInputs)
                        Inputnodes=np.delete(Inputnodes,link) 
                        InputWeights=np.delete(InputWeights,link)

            ######## mutation on number of outgoing connection
            if random.random() < mutation_rate :
                if NumOutputs==1:
                    NumOutputs=NumOutputs+1
                    OutputWeights=np.append(OutputWeights,B_max_w * np.random.uniform(-1, 1,1))

                    remain_A=[]
                    A=list(range(max_output))
                    remain_A=list(set(A)-set(Outputnodes))
                    index=random.randint(0,len(remain_A)-1)
                    node=remain_A[index]
                    Outputnodes=np.append(Outputnodes,node)


                elif NumOutputs==np.floor(max_output*con_rate):  
                    NumOutputs=NumOutputs-1
                    link=random.randint(0,NumOutputs)
                    Outputnodes=np.delete(Outputnodes,link) 
                    OutputWeights=np.delete(OutputWeights,link)

                else:
                    if random.randint(0,1)==0:
                        NumOutputs=NumOutputs+1
                        OutputWeights=np.append(OutputWeights,B_max_w * np.random.uniform(-1, 1,1))

                        remain_A=[]
                        A=list(range(max_output))
                        remain_A=list(set(A)-set(Outputnodes))
                        index=random.randint(0,len(remain_A)-1)
                        node=remain_A[index]
                        Outputnodes=np.append(Outputnodes,node)

                    else:    
                        NumOutputs=NumOutputs-1
                        link=random.randint(0,NumOutputs)
                        Outputnodes=np.delete(Outputnodes,link) 
                        OutputWeights=np.delete(OutputWeights,link)

            Parent_new=np.append(Parent_new,NumInputs ) 
            Parent_new=np.append(Parent_new,NumOutputs )
            Parent_new=np.append(Parent_new,Inputnodes )
            Parent_new=np.append(Parent_new,Outputnodes )
            Parent_new=np.append(Parent_new,InputWeights )
            Parent_new=np.append(Parent_new,OutputWeights )
        
        
        ####-----------
        OutputConnected=[]
        k=1
        for i in range(0,int(Parent_new[0])):
            NumInputs=int(Parent_new[k])
            NumOutputs=int(Parent_new[k+1])
            Outputnodes=[int(x) for x in Parent_new[k+2+NumInputs:k+2+NumInputs+NumOutputs ] ]
            OutputConnected.extend(Outputnodes)
            k=k+2 +2 *NumInputs+2* NumOutputs 
            
        if 0 in OutputConnected:
            flag=True            
    return Parent_new    
 
#-----------------------------------------------
    
def tournamentSelection(Fitness,PopSize,tournrate):
    tournsize=int(np.floor(PopSize*tournrate))
    indices = range(PopSize)
    selected_indices=random.sample(indices,tournsize)
    selected_fitness=Fitness[selected_indices]
    index_min=np.argmin(selected_fitness)
    return index_min 




##=====================================================================================
def Valid(Pop,Fitness,X,Data):
    Network=Pop[np.argmin(Fitness)]
    
    T=len(Data) 
    NumNeuron=int(Network[0])

    A=np.zeros([NumNeuron,max_input])
    B=np.zeros([max_output,NumNeuron])
    OutputConnected=[]
    k=1
    for i in range(0,NumNeuron):
        NumInputs=int(Network[k])
        NumOutputs=int(Network[k+1])
            
        Inputnodes=[int(x) for x in Network[k+2:k+2+NumInputs] ]
        Outputnodes=[int(x) for x in Network[k+2+NumInputs:k+2+NumInputs+NumOutputs ] ]
        OutputConnected.extend(Outputnodes)        
        InputWeights=Network[k+2+NumInputs+NumOutputs :k+2+ 2 *NumInputs+NumOutputs  ]
        OutputWeights=Network[k+2 + 2*NumInputs+NumOutputs :k+2 +2 *NumInputs+2* NumOutputs  ]
        k=k+2 +2 *NumInputs+2* NumOutputs
            
        A[i,Inputnodes]=InputWeights.copy()
        B[Outputnodes,i]=OutputWeights.copy()
    UnconnectedOutputs=list(set(range(0,max_input))-set(OutputConnected))
    
    x_hat=np.zeros(T)
    x_hat[0]=Data[0].copy()
    XNext=X.copy()
    XNext[0]=Data[0].copy()
            
    for t in range(1,T):
        XNext=np.dot(B,np.tanh(np.dot(A,XNext)))
        XNext[UnconnectedOutputs]=X[UnconnectedOutputs].copy()
        x_hat[t]=XNext[0].copy()
        XNext[0]=Data[t].copy()
        
    y_p=np.reshape(x_hat,[len(x_hat),1])  
    #y_p=Sc.inverse_transform(x_hat.reshape(-1,1))
    return y_p,Network    
####--------------------
def Evaluate(Pop,Fitness,X,Data,Sc):
    Network=Pop[np.argmin(Fitness)] 
    T=len(Data) 
    NumNeuron=int(Network[0])

    A=np.zeros([NumNeuron,max_input])
    B=np.zeros([max_output,NumNeuron])
    OutputConnected=[]
    k=1
    for i in range(0,NumNeuron):
        NumInputs=int(Network[k])
        NumOutputs=int(Network[k+1])
            
        Inputnodes=[int(x) for x in Network[k+2:k+2+NumInputs] ]
        Outputnodes=[int(x) for x in Network[k+2+NumInputs:k+2+NumInputs+NumOutputs ] ]
        OutputConnected.extend(Outputnodes)        
        InputWeights=Network[k+2+NumInputs+NumOutputs :k+2+ 2 *NumInputs+NumOutputs  ]
        OutputWeights=Network[k+2 + 2*NumInputs+NumOutputs :k+2 +2 *NumInputs+2* NumOutputs  ]
        k=k+2 +2 *NumInputs+2* NumOutputs
            
        A[i,Inputnodes]=InputWeights.copy()
        B[Outputnodes,i]=OutputWeights.copy()
    UnconnectedOutputs=list(set(range(0,max_input))-set(OutputConnected))
    
    x_hat=np.zeros(T)
    x_hat[0]=Data[0].copy() 
    XNext=X.copy()
    XNext[0]=Data[0].copy() 
    mse=0        
    for t in range(1,T):
        XNext=np.dot(B,np.tanh(np.dot(A,XNext)))
        XNext[UnconnectedOutputs]=X[UnconnectedOutputs].copy()
        x_hat[t]=XNext[0].copy()
        XNext[0]=Data[t].copy()
  
    y_p=Sc.inverse_transform(x_hat.reshape(-1,1))  
    return y_p,Network       
############----------------------------
def Gragh_pannet(Network,run=1):   

    dot = Digraph('unix', filename='unix.gv')
    dot.body.append('size="6,6"')
    dot.node_attr.update(color='lightblue2', style='filled')
    
    In=['I'+str(i+1) for i in range(inN)] # external input nodes
    Obs=['O'+str(i+1) for i in range(obsN)]   #observation nodes
    context=['C'+str(i+1) for i in range(max_context)]   #context nodes
    value=In+Obs+context
    key=range(obsN+max_context+inN)
    assign = dict(zip(key, value))
    
    k=1
    NumNeuron=int(Network[0])
    for neuron in range(0,NumNeuron):
        NumInputs=int(Network[k])
        NumOutputs=int(Network[k+1])
            
        Inputnodes=[int(x) for x in Network[k+2:k+2+NumInputs] ]
        Outputnodes=[int(x) for x in Network[k+2+NumInputs:k+2+NumInputs+NumOutputs ] ]
        k=k+2 +2 *NumInputs+2* NumOutputs
        
        for input in range(0,NumInputs):
            dot.edge(assign[Inputnodes[input]],'neuron'+str(neuron+1))
                
        for output in range(0,NumOutputs):
            dot.edge('neuron'+str(neuron+1),assign[Outputnodes[output]])        
    dot.render('network run'+ str(run)+'.gv')
    
########3---------------------    
def MSE(orig,predict):
    error=np.mean((orig-predict)**2)
    return error
def MAE(orig,predict):
    error=np.mean(np.abs(orig-predict))
    return error 
def R(orig,predict):## correlation cofficient
    morig=np.mean(orig)
    mpredict=np.mean(predict)
    T = len(orig)
    xm, ym = orig-morig, predict-mpredict
    r_num=np.sum(xm*ym)
    r_den = np.sqrt(np.sum(xm**2))*np.sqrt(np.sum(ym**2))
    r = r_num / r_den
    return r