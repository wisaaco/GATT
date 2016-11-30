# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:39:05 2016

@author: isaac
"""

import GATT as g
import numpy as np
import random
import time

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

     "failure":[0.0015*52,0.019*52],
     
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

     "failure":[0.0015*52,0.019*52],
     
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

     "failure":[0.0015*52,0.019*52],
     
     "CPU_idle":103,
     "CPU_full":147,
     "HD_idle":2.9,
     "MM_vendor":2
     }
]

#==============================================================================
# SAMPLES VM
#==============================================================================

casesOfVM = [
    {"id":1,"Core":4,
     "MM":7.5, #GB
     "HD":0.078125, #TB 80GB
     "failure":[0.002*52,0.025*52]
     }, 
    {"id":2,"Core":8,
     "MM":15, #GB
     "HD":0.1562, #TB 160GB
     "failure":[0.002*52,0.025*52]
     },
    {"id":3,"Core":16,
     "MM":30, #GB
     "HD":0.3125, #TB 320GB
     "failure":[0.002*52,0.025*52]},
    {"id":4,"Core":4,
     "MM":15, #GB
     "HD":0.078125, #TB 80GB
     "failure":[0.002*52,0.025*52]}  ,
    {"id":5,"Core":8,
     "MM":30, #GB
     "HD":0.1562, #TB 160GB
     "failure":[0.002*52,0.025*52]}
]

#test = { 1:[{"1":10,"2":34},30],2:[30,60],10:[10,30]}


#==============================================================================
#==============================================================================
# # Environment
#==============================================================================
#==============================================================================

environment = {
    "replicationFactor" : 3,
    "sizeBlock": 64, #MB
    
    "numberOfPM": 20, #600,
    "numberJobs": 1585 #95109 #random.randint(10,45),
}

environment["PM"] = np.random.choice(casesOfPM,environment["numberOfPM"]) #Nice: p=[.33,.33,.33]


#Distribución de uso de ficheros sobre los diferentes trabajos
alpha= 0.83
r2= 3
p = np.random.power(alpha,environment["numberJobs"])
p = np.multiply(p,r2).astype(int) #Este valor se corta a entero menor perdemos una repeticion

filesUsedInJob = []
for ix in range(environment["numberJobs"]):
    filesUsedInJob.append(
#    np.hstack(([ix],np.random.choice(environment["numberJobs"],p[ix]))) #La perticion perdida se asigna al jobs
      np.hstack(([ix]))) #La perticion perdida se asigna al jobs
#    )
    
print "Total Files con repeticion: %i" %np.sum(np.add(p,1))
print "Total Trabajos: %i " %environment["numberJobs"]


#TOTALBLOQUES = 500/environment["replicationFactor"]
# a, m  = 1.1,0.5
a, m  = 1.001,0.005
sizeFiles = []
for ix in range(environment["numberJobs"]):
    sizeFiles.append(int(np.random.pareto(a)*m)+1)
    
print "Total bloques: %i " %np.sum(sizeFiles)
print "Total bloques con Replic Factor: %i " %(np.sum(sizeFiles)*environment["replicationFactor"])
#Caracteristicas de los jobs
#Tiempo de ejecución
# mu, sigma = 96.91, 445.72# mean and standard deviation
mu, sigma = 4.32, 1.31 # mean and standard deviation
tExecution = np.random.lognormal(mu, sigma,environment["numberJobs"])
tExecution = np.multiply(tExecution,1000)

#Mismo ratio de llegada de todos los trabajos
lambda_arrivalRate= 0.44/float(environment["numberJobs"]) # por minuto
arrivalRate_job =  np.linspace(lambda_arrivalRate/environment["numberJobs"],lambda_arrivalRate/environment["numberJobs"],environment["numberJobs"]).astype(float)
arrivalRate_job = np.divide(arrivalRate_job,60000)

jobs = np.array([arrivalRate_job , #Frecuencia de llegadas
                 tExecution, #tCPU (ms) 
                 [int(random.uniform(0, 1)*10) for _ in range(environment["numberJobs"])], #Capcity MM (MB) int
                 [random.uniform(0, 1) for _ in range(environment["numberJobs"])],#tHD (ms)
                 ]).T           
                  
                 
                 
def uCPU(job):
    return job[0]* job[1] #*1000
def uMM(job):
    return job[0]* job[2]
def uHD(job):
    return job[0]* job[3]
def getU(job):
    return np.array([uCPU(job),uMM(job),uHD(job)])


#DEFINITION OF FILES 
#Update: Tan solo se definen los ficheros que se usarán
#Tamaño en bloques de cada uno de los ficheros 
#TOTALBLOQUES = 16384/environment["replicationFactor"]
#totalBloquesFile = TOTALBLOQUES/environment["numberOfJobs"]
fileU = []
for idx,value in enumerate(filesUsedInJob):
    u = np.array([0,0,0])
    for fileID in value:
        u = np.add(u,getU(jobs[idx]))
    fileU.append([row for row in u])
fileU = np.array(fileU)         
#environment["JOBS"] = jobs 
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
    f.flush()


#==============================================================================
#  #  # MAIN
#==============================================================================

mutationHappen = 0.08 #p que ocurra
# sizePopulation = 20 # different VM 20
totalGeneration = 160

clustersize="low"

maxIteration = 10




for sizePopulation in [20,100]:
    print "*********************************  Population: %i **********************" % sizePopulation
    for idx  in range(0,1):  #  == for idx,cutting in enumerate([g.C1_OneCuttingPoint,g.C2_TwoCuttingPoint]):

        print "\t IDX_Cutting: %i" % idx
        for n_iteration in range(1,maxIteration+1):
            print "\t Iteration: %i" % n_iteration
            start_time = time.time()

            gatt = g.GATT(environment, casesOfVM, seed=n_iteration)
            gatt.logger.info("[ROOT] ******** Population: %i ***  IDX Cutting: %i **** Iteration: %i " % (sizePopulation,idx,n_iteration))
            gatt.population(sizePopulation)



            # fileFitInfo = open("fits-%ic" + clustersize + "-n" + str(n_iteration) + "-s" + str(sizePopulation) + "-g" + str(totalGeneration) + ".csv", "wr")
            fileFitInfo = open("data2/fits-cross%i-c%s-s%i-g%i-n%i.csv" %((idx+1),clustersize,sizePopulation,totalGeneration,n_iteration), "wr")
            fileFitInfo.write("generation,Wmax,Wmin,Pmax,Pmin,Fmax,Fmin,Wmean,Pmean,Fmean,Max_CVM,Min_CVM,Mean_CVM,Fit_maxPareto,fitValue,Fit_meanPareto,\n")
            # fileFitInfo.write("generation,Wmax,Wmin,Pmax,Pmin,Fmax,Fmin,Wmean,Pmean,Fmean,fitValue,Max_CVM,Min_CVM,Mean_CVM,HWmax,HWmin,HPmax,HPmin,HFmax,HFmin,NP,Fit_maxPareto,Fit_meanPareto,\n")
            # fileFitInfo.write("generation,Wmax,Wmin,Pmax,Pmin,Fmax,Fmin,Wmean,Pmean,Fmean,Max_CVM,Mean_CVM,Min_CVM,Fit_maxPareto,Fit_meanPareto,fitValue,NWmax,NWmin,NPmax,NPmin,NFmax,NFmin,NWmean,NPmean,NFmean\n")

            # fileFitCit = open("fitnessCitizen.csv","w")
            # fileUPMI = open("Utilization_PM.csv","w")
            # fileProb = open("probablities_citizen.csv","w")

            saveFits(fileFitInfo,0,gatt.fitnessGeneration)

            for generation in range(1,totalGeneration):
                print "\t\tGeneration: %i" %generation
                                                    #C1_OneCuttingPoint   C2_TwoCuttingPoint
                nextGeneration = gatt.evolve(idx,0.9,tries=10000,generation=generation)
                if not nextGeneration:
                    print "ERROR - NO MORE EVOLUTION "
                    break

                if random.random() <= mutationHappen:
                    state = gatt.mutate()
                    print "\t\t\t Mutation: %s" %state

                #Fit values
                saveFits(fileFitInfo,generation,gatt.fitnessGeneration)

                #Prob values
                # saveFits(fileProb, generation, gatt.probFitness)

                #Fitness citizen
                # saveFits(fileFitCit, generation, gatt.fit)

                #U PMI
                #print gatt.getUPMI(0)
                # saveFits(fileUPMI, generation,gatt.getUPMI(0))
                # print("\t\t- %s seconds ---" % (time.time() - start_time))

            fileFitInfo.close()
            # fileFitCit.close()
            # fileProb.close()
            # fileUPMI.close()
            gatt.logger.info("TOTAL TIME: %s seconds ---" % (time.time() - start_time))
            print("TOTAL TIME: %s seconds ---" % (time.time() - start_time))