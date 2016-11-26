# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:39:05 2016

@author: isaac
"""

import GATT as g
import numpy as np
import random

import matplotlib.pyplot as plt


random.seed(100)
np.random.seed(100)

#==============================================================================
# SAMPLES PM 
#==============================================================================

casesOfPM = [
    {"ID" :"R730",
     
    "Core":24,
     "MM":768, #GB
     "HD":8, #TB

     "XHD":1,
     "tresholdCPU":0.85,
     "tresholdHD": 0.6,
     "tresholdMM":0.75,

     "failure":[0.0015,0.019],
     
     "CPU_idle":124,
     "CPU_full":265,
     "HD_idle":5.7,
     "MM_vendor":2
     },
     #tasa de fallos
    {
    "ID":"R430",
    
    "Core":12,
     "MM":384, #GB
     "HD":5, #TB
     
     "XHD":1,
     "tresholdCPU":0.85,
     "tresholdHD": 0.6,
     "tresholdMM":0.75,

     "failure":[0.0015,0.019],
     
     "CPU_idle":142,
     "CPU_full":284,
     "HD_idle":4.9,
     "MM_vendor":2
     },
     
    {
    "ID":"R330",
    
    "Core":4,
     "MM":64, #GB
     "HD":1, #TB
     
     "XHD":1,
     "tresholdCPU":0.85,
     "tresholdHD": 0.6,
     "tresholdMM":0.75,

     "failure":[0.0015,0.019],
     
     "CPU_idle":103,
     "CPU_full":147,
     "HD_idle":2.9,
     "MM_vendor":2
     }
]

#==============================================================================
# SAMPLES VM
#==============================================================================
#casesOfVM = [
#    {"id":1,"Core":1,
#     "MM":2, #GB
#     "HD":1, #TB 
#     "failure":0.1
#     }, 
#    {"id":2,"Core":3,
#     "MM":4, #GB
#     "HD":2, #TB
#     "failure":0.1},
#    {"id":3,"Core":2,
#     "MM":6, #GB
#     "HD":4, #TB
#     "failure":0.1}
#]

casesOfVM = [
    {"id":1,"Core":4,
     "MM":7.5, #GB
     "HD":0.078125, #TB 80GB
     "failure":[0.002,0.025]
     }, 
    {"id":2,"Core":8,
     "MM":15, #GB
     "HD":0.1562, #TB 160GB
     "failure":[0.002,0.025]
     },
    {"id":3,"Core":16,
     "MM":30, #GB
     "HD":0.3125, #TB 320GB
     "failure":[0.002,0.025]},
    {"id":4,"Core":4,
     "MM":15, #GB
     "HD":0.078125, #TB 80GB
     "failure":[0.002,0.025]}  ,
    {"id":5,"Core":8,
     "MM":30, #GB
     "HD":0.1562, #TB 160GB
     "failure":[0.002,0.025]}
]

test = { "1":[10,30],"2":[30,60],"3":[10,30]}


#==============================================================================
#==============================================================================
# # Environment
#==============================================================================
#==============================================================================

environment = {
    "replicationFactor" : 3,
    "sizeBlock": 64, #MB
    
    "numberOfPM": 20,
    "numberFile": random.randint(10,45),
#    "numberOfJobs": #random.randint(10,20) 
  
}
environment["numberOfJobs"]=environment["numberFile"] #TODO BORRAR FUTURO

environment["PM"] = np.random.choice(casesOfPM,environment["numberOfPM"]) #Nice: p=[.33,.33,.33]

#DEFINITION OF JOBS
# Jobs 
#fjobs = np.random.choice(environment["numberFile"],environment["numberOfJobs"]) #TODO AGGGRE FUTURO
fjobs = np.array(range(environment["numberOfJobs"]))


mu, sigma = 4.32, 1.31 # mean and standard deviation
tExecution = np.random.lognormal(mu, sigma,environment["numberOfJobs"])
tExecution = np.multiply(tExecution,1000)
jobs = np.array([np.random.uniform(0,1,environment["numberOfJobs"]), #Frecuencia de llegadas
                 tExecution, #tCPU (ms) 
                 [int(random.uniform(0, 1)*10) for _ in range(environment["numberOfJobs"])], #Capcity MM (MB) int
                 [random.uniform(0, 1) for _ in range(environment["numberOfJobs"])],#tHD (ms)
                 fjobs, #ID de files
                 ]).T

jobArrivalRate = 455/60000.0                
seqAR =  np.linspace(jobArrivalRate/environment["numberOfJobs"],jobArrivalRate/environment["numberOfJobs"],environment["numberOfJobs"]).astype(float)



jobs = np.array([seqAR , #Frecuencia de llegadas
                 [random.uniform(0, 1) for _ in range(environment["numberOfJobs"])], #tCPU (ms) 
                 [int(random.uniform(0, 1)*10) for _ in range(environment["numberOfJobs"])], #Capcity MM (MB) int
                 [random.uniform(0, 1) for _ in range(environment["numberOfJobs"])],#tHD (ms)
                 fjobs, #ID de files
                 ]).T           
                     
                 
                 
def uCPU(job):
    return job[0]* job[1]
def uMM(job):
    return job[0]* job[2]
def uHD(job):
    return job[0]* job[3]
def getU(job):
    return np.array([uCPU(job),uMM(job),uHD(job)])


#DEFINITION OF FILES 
#Update: Tan solo se definen los ficheros que se usarán
#Tamaño en bloques de cada uno de los ficheros 
TOTALBLOQUES = 16384/environment["replicationFactor"]
TOTALBLOQUES = 500/environment["replicationFactor"]
totalBloquesFile = TOTALBLOQUES/environment["numberOfJobs"]


rjobs = np.bincount(fjobs) #in order
fileU = range(len(fjobs))
sizeFiles = range(len(fjobs))
for idx,value in enumerate(rjobs):
    if (value!=0):
        ids = np.where(fjobs==idx) #Donde aparece el fichero en JOB
        u = np.array([0,0,0])
#        size = random.randint(1,10) #TODO 
        size = totalBloquesFile
        for i in ids[0]: #Para cada JOB se calcula la utilización
            u = np.add(u,getU(jobs[i]))
        for i in ids[0]: #Para cada aparición se inserta la utilización
            fileU[i] = [row for row in u]
            sizeFiles[i]=size
            
fileU = np.array(fileU)         
environment["JOBS"] = jobs 
environment["sizeFiles"] = sizeFiles
environment["fileU"] = fileU

#==============================================================================
# 
#==============================================================================
def saveFits(f,generation,fitInfo):
    f.write("%i"%generation)
    for v in fitInfo:
        f.write(",%f"%v)
    f.write("\n")
#==============================================================================
# 
#==============================================================================

mutationHappen = 0.08 #p que ocurra
sizePopulation = 100 # different VM
gatt = g.GATT(environment,casesOfVM,seed=100)

gatt.population(sizePopulation)
print len(gatt.pop)

fitMin = []
with open("fits.csv","wr") as f:
    f.write("generation, Wmax,Wmin,Pmax,Pmin,Fmax,Fmin,Wmean,Pmean,Fmean,fitValue\n")
# [Wmax,Wmin,Pmax,Pmin,Fmax,Fmin,Wmean,Pmean,Fmean,fitValue]
    fitInfo = gatt.fitnessGeneration
    print "FIT  Info %s" %fitInfo
    saveFits(f,0,fitInfo)
    fitMin.append(fitInfo[9])

    for generation in range(1,100):
        print "Generation: %i" %generation
    
        nextGeneration = gatt.evolve(gatt.C1_OneCuttingPoint,0.9,tries=10000)
        if not nextGeneration:
            print "No evolution"
            break
        
        if random.random() <= mutationHappen:
            state = gatt.mutate()
            print "\t Mutation: %s" %state   
    
        #Fit values
        fitInfo = gatt.fitnessGeneration
        saveFits(f,generation,fitInfo)
        f.flush()
        fitMin.append(fitInfo[9])

# np.save("fitMin.npy",fitMin)
# fitMin = np.load("fitMin.npy")
plt.plot(fitMin)
plt.show()

#gatt.show()

