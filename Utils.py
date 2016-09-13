import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from graphviz import Digraph
import pydotplus

#---------------------------------------------------------------------------------------------

def initialization():
    global PopSize,NumGen,max_neurons,min_neurons,max_context,obsN,inN,max_input,max_output,con_rate
    global crossover_rate,mutation_rate,elit_rate,tournrate,B_max_w,w_rate,X0
    PopSize=100
    NumGen=2000
    max_neurons=8      #maximum number of neurons
    min_neurons=4
    max_context=7
    obsN=1          # dimension of time series
    inN=0            # external input
    max_input=max_context+obsN+inN     #maximum number of inputs for neurons
    max_output=max_context+obsN        #maximum number of outputs for neurons
    con_rate=1                       # ratio of number nodes that should be connected to neurons

    B_max_w=.5# connection weight boundary
    w_rate=0.01
    mutation_rate =0.01 #mutate input nodes, output nides, weights 
    elit_rate=0.1
    crossover_rate=.5
    tournrate=0.1
    X0=0.1*np.ones(max_input)#np.random.uniform(-.01, .01,max_input)
    
    #StrucParams=[max_neurons,min_neurons,max_context,obsN,inN,B_max_w,con_rate,max_input,max_output]

    
    #return StrucParams,PopSize,NumGen,w_rate,mutation_rate,elit_rate,crossover_rate,tournrate,X0
    #return PopSize,
    ##---------------------------------------------------------------------------------------------------------
def Evolution(Data,y_a,n_training,Sc):
    initialization()
    
    TrainingData=Data[:n_training]
    Train_orig=y_a[:n_training]  #y_a
    Test_orig=y_a[n_training:]#Sunspot_normed
    
    MinFit=np.empty(NumGen)
    MaxFit=np.empty(NumGen)
    AvgFit=np.empty(NumGen)
    UNI=np.empty(NumGen)


    Predicted_output=[]
    Network=[]


    Pop=[GenerateIndividual() for i in range(0,PopSize)]


    Fitness=np.empty(PopSize)
    for i in range(0,PopSize):
        Fitness[i]=CalcFitness(Pop[i],TrainingData,X0)
    #Fitness=np.array([CalcFitness(i,TrainingData,X0) for i in Pop])
    ######------
    Error=pd.DataFrame(columns=['Train MSE','Train MAE','Test MSE','Test MAE'],index=range(NumGen))

    #c=10**10
    for j in range(0,int(NumGen/2)):
        ind_best=Fitness.argsort()[:int(np.floor(elit_rate*PopSize))].copy()
        elit_ind=[Pop[int(a)] for a in ind_best].copy()

        selectedpop=[Pop[tournamentSelection(Fitness,PopSize,tournrate)] for i in range(0,PopSize)]
    ###########----------------
        crosspop=[]
        k=range(0,PopSize)
        k=random.sample(k,PopSize)

        for i in range(0,int(PopSize/2)):  
            if random.random() < crossover_rate: # rate dependent crossover of selected chromosomes
                CrossChild1,CrossChild2= Crossover(selectedpop[k[2*i]],selectedpop[k[2*i+1]],crossover_rate)
            else:
                CrossChild1=selectedpop[k[2*i]]
                CrossChild2=selectedpop[k[2*i+1]]
            crosspop.append(CrossChild1)
            crosspop.append(CrossChild2)

    ########-----------
        newpop=[]
        for i in range(0,PopSize):
            newpop.append(Mutation(crosspop[i].copy(),mutation_rate,w_rate))

            #stopm = timeit.default_timer()
            #print('muttime=',stopm-startm)        
        newpop.extend(elit_ind)

        #######-----------
        newsize=PopSize+int(np.floor(elit_rate*PopSize))

        newFitness=np.empty(newsize)
        for i in range(0,newsize):
            newFitness[i]=CalcFitness(newpop[i],TrainingData,X0)

        #newFitness=np.array([CalcFitness(i,TrainingData,X0) for i in newpop])

        index_fittest=newFitness.argsort()[:PopSize].copy()
        Pop=[newpop[int(a)] for a in index_fittest] .copy()  
        Fitness=newFitness[index_fittest].copy()

        ######----------
        y_p,network=Evaluate(Pop,Fitness,X0,Data,Sc)
        Predicted_output.append(y_p)
        Network.append(network)

        #############------------------------------------
        Train_predict=y_p[:n_training] #.copy()
        Test_predict=y_p[n_training:]  #.copy()

        Error['Train MAE'].iloc[j]=MAE(Train_orig,Train_predict) 
        Error['Test MAE'].iloc[j]=MAE(Test_orig,Test_predict)

        Error['Train MSE'].iloc[j]=MSE(Train_orig,Train_predict)   #Train_orig float("{0:.6f}".format(x)
        Error['Test MSE'].iloc[j]=MSE(Test_orig,Test_predict)

        ########-------------
        MinFit[j]=min(Fitness)
        MaxFit[j]=max(Fitness)
        AvgFit[j]=np.mean(Fitness)
        UNI[j]=len(np.unique(Fitness))
    ###########################################################-------------- mutation on weight
    for j in range(int(NumGen/2),NumGen):
        ind_best=Fitness.argsort()[:int(np.floor(elit_rate*PopSize))].copy()
        elit_ind=[Pop[int(a)] for a in ind_best].copy()

        selectedpop=[Pop[tournamentSelection(Fitness,PopSize,tournrate)] for i in range(0,PopSize)]
    ###########----------------
        crosspop=[]
        k=range(0,PopSize)
        k=random.sample(k,PopSize)

        for i in range(0,int(PopSize/2)):  
            if random.random() < crossover_rate: # rate dependent crossover of selected chromosomes
                CrossChild1,CrossChild2= Crossover(selectedpop[k[2*i]],selectedpop[k[2*i+1]],crossover_rate)
            else:
                CrossChild1=selectedpop[k[2*i]]
                CrossChild2=selectedpop[k[2*i+1]]
            crosspop.append(CrossChild1)
            crosspop.append(CrossChild2)

    ########-----------
        newpop=[]
        for i in range(0,PopSize):
            newpop.append(WMutation(crosspop[i].copy(),mutation_rate,w_rate))

            #stopm = timeit.default_timer()
            #print('muttime=',stopm-startm)        
        newpop.extend(elit_ind)

        #######-----------
        newsize=PopSize+int(np.floor(elit_rate*PopSize))

        newFitness=np.empty(newsize)
        for i in range(0,newsize):
            newFitness[i]=CalcFitness(newpop[i],TrainingData,X0)

        #newFitness=np.array([CalcFitness(i,TrainingData,X0) for i in newpop])

        index_fittest=newFitness.argsort()[:PopSize].copy()
        Pop=[newpop[int(a)] for a in index_fittest] .copy()  
        Fitness=newFitness[index_fittest].copy()

        ######----------
        y_p,network=Evaluate(Pop,Fitness,X0,Data,Sc)
        Predicted_output.append(y_p)
        Network.append(network)

        #############------------------------------------
        Train_predict=y_p[:n_training] #.copy()
        Test_predict=y_p[n_training:]  #.copy()

        Error['Train MAE'].iloc[j]=MAE(Train_orig,Train_predict) 
        Error['Test MAE'].iloc[j]=MAE(Test_orig,Test_predict)

        Error['Train MSE'].iloc[j]=MSE(Train_orig,Train_predict)   #Train_orig float("{0:.6f}".format(x)
        Error['Test MSE'].iloc[j]=MSE(Test_orig,Test_predict)

        ########-------------
        MinFit[j]=min(Fitness)
        MaxFit[j]=max(Fitness)
        AvgFit[j]=np.mean(Fitness)
        UNI[j]=len(np.unique(Fitness))
    return Error, MinFit,MaxFit,AvgFit,UNI,Train_predict,Test_predict,Network
  
    
###-------------------------------    
def GenerateIndividual():
    #max_neurons,min_neurons=n[0],n[1]
    #inN,B_max_w,con_rate,max_input,max_output=n[4],n[5],n[6],n[7],n[8]
    #initialization()
    """
    NumNeurons=random.randint(min_neurons,max_neurons)
    Individual=np.array([NumNeurons])
    for i in range(NumNeurons):
        NumInputs=random.randint(1,np.floor(max_input*con_rate))
        NumOutputs=random.randint(1,np.floor(max_output*con_rate))
        InputIndices=random.sample(range(0, max_input-1), NumInputs)
        OutputIndices=random.sample(range(inN, max_output-inN-1), NumOutputs)  #don't connect inN to output of neurons
        WeightInput=B_max_w * np.random.uniform(-1, 1,NumInputs)
        WeightOutput=B_max_w * np.random.uniform(-1, 1,NumOutputs)
        Individual=np.hstack( [Individual,NumInputs,NumOutputs,InputIndices,OutputIndices,WeightInput,WeightOutput])

    """
    x=False
    while x==False:
        NumNeurons=random.randint(min_neurons,max_neurons)
        Individual=np.array([NumNeurons])
        check=[]
        for i in range(NumNeurons):
            NumInputs=random.randint(1,np.floor(max_input*con_rate))
            NumOutputs=random.randint(1,np.floor(max_output*con_rate))
            InputIndices=random.sample(range(0, max_input), NumInputs)
            OutputIndices=random.sample(range(inN, max_input), NumOutputs)  #don't connect inN to output of neurons
            check.extend(OutputIndices)
            WeightInput=B_max_w * np.random.uniform(-1, 1,NumInputs)
            WeightOutput=B_max_w * np.random.uniform(-1, 1,NumOutputs)
            Individual=np.hstack( [Individual,NumInputs,NumOutputs,InputIndices,OutputIndices,WeightInput,WeightOutput])
        if 0 in check:
            x=True
    
    return Individual


    #----------------------------------------------------------------------------------------------------

def CalcFitness(Individual,TrainingData,X):
    #obsN=n[3]
    #inN,max_input,max_output=n[4],n[7],n[8]
    #initialization()
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
        K=K+NumInputs+NumOutputs ## number of connections
        k=k+2 +2 *NumInputs+2* NumOutputs
        
        A[i,Inputnodes]=InputWeights
        B[Outputnodes,i]=OutputWeights

    UnconnectedOutputs=list(set(range(0,max_input))-set(OutputConnected))    
    
    Fitness=100
    #x_hat_fit=np.zeros(T)#np.reshape(,[T,1])
    #x_hat_fit[0]=TrainingData[0].copy() 
    if 0 not in UnconnectedOutputs:
        
        
        XNext=X.copy()
        a=TrainingData[0]
        XNext[0]=a.copy()
        
        Fitness=0

        for t in range(1,T):

            XNext=np.dot(B,np.tanh(np.dot(A,XNext)))
            XNext[UnconnectedOutputs]=X[UnconnectedOutputs].copy()
            x_hat=XNext[0].copy()
            #x_hat_fit[t]=x_hat.copy()
            Fitness=Fitness +  (x_hat-TrainingData[t][0])**2      #[0] is added
            a=TrainingData[t]
            XNext[0]=a.copy()
        Fitness=Fitness/T

    #print(Fitness,0.01*s_w  )
    Fitnes=Fitness+0.1*s_w  
    
    #Fitness= T*np.log(Fitness)+2*K  ### AIC
    #AICc=AIC+2*K*(K+1)/(time-K-1)
    #if Fitness==100:
       # print('e')
    return Fitness    #,M

    ##---------------------------------------------------------------------------------------------------------

def Crossover(Parent1,Parent2,crossover_rate):
    #initialization()
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
def Mutation(Parent,mutation_rate,w_rate):
    #max_neurons,min_neurons,obsN=n[0],n[1],n[3]
    #inN,B_max_w,con_rate,max_input,max_output=n[4],n[5],n[6],n[7],n[8]
    #initialization()
    flag=False
    while flag==False:
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
        for neuron in range(0,NumNeurons): #
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


            #-----------------------------------------------

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
    #initialization()
    tournsize=int(np.floor(PopSize*tournrate))
    indices = range(PopSize)
    selected_indices=random.sample(indices,tournsize)
    selected_fitness=Fitness[selected_indices]
    index_min=np.argmin(selected_fitness)
    return index_min 




##=====================================================================================
def valid(Pop,Fitness,X,Data):
    #initialization()
    Network=Pop[np.argmin(Fitness)]
    
    T=len(Data) 
    NumNeuron=int(Network[0])

    A=np.zeros([NumNeuron,max_input])
    B=np.zeros([max_output,NumNeuron])

    k=1
    OutputConnected=[]
    for i in range(0,NumNeuron):
        NumInputs=int(Network[k])
        NumOutputs=int(Network[k+1])
            
        Inputnodes=[int(x) for x in Network[k+2:k+2+NumInputs] ]
        Outputnodes=[int(x) for x in Network[k+2+NumInputs:k+2+NumInputs+NumOutputs ] ]
        OutputConnected.extend(Outputnodes)        
        InputWeights=Network[k+2+NumInputs+NumOutputs :k+2+ 2 *NumInputs+NumOutputs  ]
        OutputWeights=Network[k+2 + 2*NumInputs+NumOutputs :k+2 +2 *NumInputs+2* NumOutputs  ]
        k=k+2 +2 *NumInputs+2* NumOutputs
            
        A[i,Inputnodes]=InputWeights
        B[Outputnodes,i]=OutputWeights
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

    return y_p    
####--------------------
def Evaluate(Pop,Fitness,X,Data,Sc):
    #obsN=n[3]
    #inN,max_input,max_output=n[4],n[7],n[8]
    #initialization()
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
    
    x_hat=np.zeros(T)#np.reshape(,[T,1])
    x_hat[0]=Data[0].copy() 
    XNext=X.copy()
    #XNext=np.reshape(X,[max_input,1])
    XNext[0]=Data[0].copy() 
    mse=0        
    for t in range(1,T):
        XNext=np.dot(B,np.tanh(np.dot(A,XNext)))
        XNext[UnconnectedOutputs]=X[UnconnectedOutputs].copy()
        #print(XNext)
        a=XNext[0].copy()
        x_hat[t]=a
        XNext[0]=Data[t].copy()
        #if t<221:
            #mse=mse +  (x_hat[t]-Data[t][0])**2  
    #mse=mse/221    
    #print(np.shape(x_hat))    
    y_p=Sc.inverse_transform(x_hat.reshape(-1,1))  
    #fit=CalcFitness(Network,Data[:221],X) 
    return y_p,Network #,fit,mse   #y_p         
############----------------------------
def Gragh_pannet(Network,run=1):   
    #max_context,obsN=n[2],n[3]
    #inN=n[4]
    
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
    #print(dot.source)
    dot.render('network run'+ str(run)+'.gv')
    
########3---------------------    
def MSE(orig,predict):
    #error=np.sum((orig-predict)**2)/len(orig)
    #error= np.mean(np.transpose(orig)-np.transpose(predict))**2
    error=np.mean((orig-predict)**2)
    return error
def MAE(orig,predict):
    error=np.mean(np.abs(orig-predict))
    return error 
