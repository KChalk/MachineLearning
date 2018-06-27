
# coding: utf-8

# # Final project
# 
# Kendra Chalkley
# 
# CS 559 Machine Learning
# 
# June 26, 2018
# 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


inputs= pd.read_csv('medoutput.csv')
subset_inputs= pd.read_csv('subset_medoutput.csv')

freqs=inputs.loc[:,list(set(inputs.columns) - set(('subreddit','count(1)','sum(wordcount)')))]
subset_freqs=subset_inputs.loc[:,list(set(inputs.columns) - set(('subreddit','count(1)','sum(wordcount)')))]


# ### Preprocessing


def normalize(vector):
    v2=vector
    vmin=float(min(vector))
    vmax=float(max(vector))
    vrange=vmax-vmin

#    print('vector', vector, vrange)
    
    l=len(vector)
    for i in range(0,l):
        dif=float(vector[i])-vmin
#        print('dif',dif, vrange, dif/vrange)
        v2[i]=dif/vrange
#        print(vector[i])
    return v2

def normalizeall(df):
    df2=df
    i=df.shape[1]
    for j in range(0,i):
        df2[:,j]=normalize(df2[:,j])
    return df2

nfreqs=normalizeall(np.array(freqs))
nsubset_freqs=normalizeall(np.array(subset_freqs))


# ## The Algorithm
# 
# ### Distance measures: Points and Clusters



def pointDistance (point1, point2):
    dimNum=len(point1)
    if dimNum!=len(point2):
        print("we have a dimensionality problem, Houston")
        print(dimNum,len(point2))       

    dif=point1[:-1]-point2[:-1]
    s=sum(dif.T*dif)
    ssqrt=np.sqrt(s)
    #print('qrt2',ssqrt)
    return ssqrt
 
def findCenter(points):
    n=np.shape(points)[0]
    d=np.shape(points)[1]
    center=np.zeros(d)
    for i in range(0,d):
        center[i]= np.mean(points.iloc[:,i])
    return center


# ### Distance Matrix
# 


data=nsubset_freqs.copy()
n=np.shape(data)[0]
d=np.shape(data)[1]

df=pd.DataFrame(data)
df['cluster']=df.index
centers=df

distanceMatrix=np.zeros((n,n))

for i in range(0,n):
    pointa = centers.iloc[i,]
    for j in range(0,n):
        if j<=i:
            distanceMatrix[i][j]=None
        else:
            pointb = centers.iloc[j,] 
            distanceMatrix[i][j]=pointDistance(pointa, pointb)
            #print(i,j,distanceMatrix[i][j])

matrixCopy=distanceMatrix.copy()            
history=pd.DataFrame({'round1': df['cluster']})


# ### Merging Groups and Recording History
# 
#distanceMatrix=matrixCopy.copy()
#data=nsubset_freqs.copy()
df=pd.DataFrame(data)
df['cluster']=df.index
centers=df.copy()
merges=pd.DataFrame(np.zeros((n,3)),columns=['head', 'leaves', 'groupsize'])

iternum=0
while np.nanmin(distanceMatrix)>0:
    minimum=np.nanmin(distanceMatrix)
    
# get indexes where distance is minimum
    mask=np.isin(distanceMatrix, minimum)
    a,b =np.where(mask)
# label cluster with smaller label groupa    
    groupa=min(a[0],b[0])
    groupb=max(a[0],b[0])
    
#Find indexes of points which were in group b.
    mask=np.isin(df['cluster'], groupb )
    index=np.where(mask)

#Reassign them to group a    
    df.loc[mask,'cluster']=int(df.loc[groupa,'cluster'])
    
#Find points in freshly minted group a
    mask=np.isin(df['cluster'], groupa)
    groupPoints= df[mask]

#Update the cluster center of group a
    centers.iloc[groupa]=findCenter(groupPoints)
#Remove center of group b    
    centers.iloc[groupb]=None

#Save some information to reconstruct hierarchy
    merges.loc[iternum,'head']=groupa
    merges.loc[iternum,'leaves']=groupb
    merges.loc[iternum,'groupsize']=len(groupPoints)

    history[str(iternum)]=df['cluster']
#Remove group b from distance matrix
    distanceMatrix[groupb,:]=None
    distanceMatrix[:,groupb]=None
    
#Update Distances to groupa          
    i=groupa
    pointa = centers.iloc[i]
    for j in range(0,n):
        if j>groupa:
            pointb = centers.iloc[j,] 
            distanceMatrix[i][j]=pointDistance(pointa, pointb)

    iternum+=1



history.index=subset_inputs['subreddit']

merges.groupby('head').count().sort_values('leaves', ascending=False).head(10)


# # Graphs
# 


from sklearn.decomposition import PCA

pca=PCA(n_components=11)
pcs=pca.fit_transform(subset_freqs)
plotData=pd.DataFrame(pcs, index=subset_inputs['subreddit'])

plots=[]
j=0

fig = plt.figure(figsize=(14,28))


for i in range(0,639, 50):
    plotData['cluster']=history.loc[:,str(i)]
    j+=1
    fig.add_subplot(10, 2, j)
    plt.scatter(x=plotData[0],y=plotData[1], c=plotData['cluster'])
    plt.colorbar()

plt.show()
plt.savefig('scatter1.png')




fig = plt.figure(figsize=(14,28))

j=0
for i in [0,150,200,250,300,380,390,400,460,480,520,600]:
    plotData=pd.DataFrame(pcs, index=subset_inputs['subreddit'])
    plotData['cluster']=history.loc[:,str(i)]
    mask=np.isin(plotData['cluster'], [12,0,5,16,4,140,1,26,111,8])
    plotData=plotData.loc[mask]
    j+=1
    fig.add_subplot(9, 3, j)
    plt.scatter(x=plotData[0],y=plotData[1], label=plotData.index, c=plotData['cluster'].astype(object), cmap='Paired')
    plt.colorbar()


plt.show()
plt.savefig('scatter2.png')



fig = plt.figure(figsize=(14,28))

j=0
for i in [0,150,200,250,300,380,390,400,460,480,520,600,610,625,636,645]:
    plotData=pd.DataFrame(pcs, index=subset_inputs['subreddit'])
    j+=1
    fig.add_subplot(9, 3, j)

    plotData['cluster']=history.loc[:,str(i)]

    mask=np.isin(plotData['cluster'], [8])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='pink')

    mask=np.isin(plotData['cluster'], [12])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='orange')

    mask=np.isin(plotData['cluster'], [5])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='red')

    mask=np.isin(plotData['cluster'], [16])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='green')
    
    mask=np.isin(plotData['cluster'], [4])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='yellow')
    
    mask=np.isin(plotData['cluster'], [140])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='blue')
    
    mask=np.isin(plotData['cluster'], [1])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='purple')
    
    mask=np.isin(plotData['cluster'], [26])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='gray')

    mask=np.isin(plotData['cluster'], [0])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='navy')

    mask=np.isin(plotData['cluster'], [111])
    plotSubsetData=plotData.loc[mask]
    plt.scatter(x=plotSubsetData[0],y=plotSubsetData[1], label=plotData.index, c='black')
    
    
plt.show()
plt.savefig('scatter3.png')


