# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:07:09 2016

@author: isaac
"""

import GATT as g
import numpy as np
import random

random.seed(100)
np.random.seed(100)

#==============================================================================
# SAMPLES PM 
#==============================================================================

casesOfPM = [
    {"ID" :"R730",
     
    "Core":4,
     "MM":16, #GB
     "HD":10, #TB
     "XHD":1,
     "tresholdCPU":0.8,
     "tresholdHD": 0.6,
     "tresholdMM":0.9,
     "failure":0.1,
     
     "CPU_idle":124,
     "CPU_full":265,
     "HD_idle":5.7,
     "MM_vendor":2
     },
     #tasa de fallos
    {
    "ID":"R330",
    
    "Core":8,
     "MM":32, #GB
     "HD":10, #TB
     "XHD":1,
     "tresholdCPU":0.8,
     "tresholdHD": 0.6,
     "tresholdMM":0.9,
     "failure":0.1,
     
     "CPU_idle":142,
     "CPU_full":284,
     "HD_idle":4.9,
     "MM_vendor":2
     },
    {
    "ID":"R330",
    
    "Core":12,
     "MM":32, #GB
     "HD":100, #TB
     "XHD":1,
     "tresholdCPU":0.8,
     "tresholdHD": 0.6,
     "tresholdMM":0.9,
     "failure":0.1,
     
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
    {"id":1,"Core":1,
     "MM":2, #GB
     "HD":1, #TB 
     "failure":0.1
     }, 
    {"id":2,"Core":3,
     "MM":4, #GB
     "HD":2, #TB
     "failure":0.1},
    {"id":3,"Core":2,
     "MM":6, #GB
     "HD":4, #TB
     "failure":0.1}
]


#==============================================================================
#==============================================================================
# # Environment
#==============================================================================
#==============================================================================

environment = {
    "replicationFactor" : 3,
    "sizeBlock": 64, #MB
    
    "numberOfPM": 10,
    "numberFile": random.randint(10,45),
    "numberOfJobs": random.randint(10,20) 
  
}

environment["PM"] = np.random.choice(casesOfPM,environment["numberOfPM"]) #Nice: p=[.33,.33,.33]

#DEFINITION OF JOBS
# Jobs 
fjobs = np.random.choice(environment["numberFile"],environment["numberOfJobs"])

jobs = np.array([np.random.uniform(0,1,environment["numberOfJobs"]), #Frecuencia de llegadas
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
rjobs = np.bincount(fjobs) #in order
fileU = range(len(fjobs))
sizeFiles = range(len(fjobs))
for idx,value in enumerate(rjobs):
    if (value!=0):
        ids = np.where(fjobs==idx) #Donde aparece el fichero en JOB
        u = np.array([0,0,0])
        size = random.randint(1,10)
        for i in ids[0]: #Para cada JOB se calcula la utilización
            u = np.add(u,getU(jobs[i]))
        for i in ids[0]: #Para cada aparición se inserta la utilización
            fileU[i] = [row for row in u]
            sizeFiles[i]=size
fileU = np.array(fileU)         
environment["JOBS"] = jobs 
environment["sizeFiles"] = sizeFiles
environment["fileU"] = fileU


def terasToBlock(teras):
    return (teras*1024*1024)/environment["sizeBlock"]
def blockToTeras(block):
    return (block*environment["sizeBlock"])/1024/1024
def MbToTera(mb):
    return mb/1024/1024 
def MbToGB(mb):
    return mb/1024

def getTotalCPU():
    return np.sum(fileU[:,0])
def getTotalMM():
    return MbToGB(np.sum(fileU[:,1]))
def getTotalHD():
    return np.sum(fileU[:,2])


#==============================================================================
# PRINT INFO 
#==============================================================================

#PM
print "PM"
for ipm in environment["PM"]:
    print ipm 
#JOBS
print "JOBS: %i %i" %(len(environment["JOBS"]),environment["numberOfJobs"])
print "[ArrivalFreq,tCPU, CMM,tHD, IDFile]"
for job in environment["JOBS"]: 
    print job
#    print "UCPU: %s, UMM: %s, UHD: %s\n" %getU(job)
#FILES
#Listado de Files usados 
sizeFiles = environment["sizeFiles"]
totalBlocks = np.sum(sizeFiles)*environment["replicationFactor"]

print "NumberOfBlocks for each file: %s " %sizeFiles
print "Total Blocks with Replications: %i " %(totalBlocks)


totalCapacity = 0
for ipm in environment["PM"]:
    totalCapacity += ipm["HD"]
print "Total HD-capacity: %i GB (%i blocks)" %(totalCapacity,terasToBlock(totalCapacity))
print "Utilization of HD-capacity: %0.2f %% " %((totalBlocks/terasToBlock(totalCapacity)*100))
#######

#==============================================================================
# TESTING CODE
#==============================================================================

#OBTAIN A ENOUGH NUMBER OF VM
def __getDemandJobs():
    sizeFiles = environment["sizeFiles"]
    totalBlocksWR = np.sum(sizeFiles)*environment["replicationFactor"]
    totalTeraBlocks = blockToTeras(totalBlocksWR)
    tCPU = getTotalCPU()
    tMM = getTotalMM()
    return tCPU,tMM,totalTeraBlocks    
    
def __getMaxSetVM(setVM):
   tCores = 0
   tHD = 0
   tMM = 0
   for ipm in setVM:
        tHD += ipm["HD"] 
        tCores += ipm["Core"]
        tMM += ipm["MM"]
   return tCores,tMM,tHD
def __getMaxAVM(aVM):
   tCores = 0
   tHD = 0
   tMM = 0
   tHD += aVM["HD"] 
   tCores += aVM["Core"]
   tMM += aVM["MM"]
   return tCores,tMM,tHD
    
def __isSuitable(setVM):
   tC,tMM,tHD = __getMaxSetVM(setVM)
   jC,jMM,jBlocks = __getDemandJobs()
   if tC<jC or tMM < jMM or tHD < jBlocks:  return False
   return True    
   
numberVM = environment["replicationFactor"]
setVM = np.random.choice(casesOfVM,numberVM)
while not __isSuitable(setVM):
    numberVM +=1
    setVM = np.random.choice(casesOfVM,numberVM)
print "Numero de máquinas virtuales necesarias: %i " %numberVM
print "Caracteristicas: %s " %setVM


##TESTING DISTRIBUCION DE BLOQUES SOBRE VM
def cbl_availability(idVM,setVM,cbl,cbl_idf,idFile):
    #La capacidad de la máquina
    tC,tMM,tHD = __getMaxAVM(setVM[idVM])
    #La tarea influye 1/3 en el peso de la máquina
    u = environment["fileU"][idFile]/environment["replicationFactor"]
    appearsVM = np.where(cbl==idVM) #Cuantas veces esta asignada esa VM
    filesAso = []
    uVM = u #La uAssignada más la utilización futura 
    for ap in appearsVM[0]:
        #Que files tiene asociados
        ut = environment["fileU"][cbl_idf[ap]]/environment["replicationFactor"]
        uVM = np.add(ut,uVM)
    #La nueva Uassginada es factible en esa VM?
    if uVM[0]>tC: 
#        print "Core"
        return False
    if MbToGB(uVM[1])>tMM: 
#        print "MM"
        return False   
    if uVM[2]>tHD: 
#        print "HD"
        return False   
    return True

setVM = np.random.choice(casesOfVM,4)
sizeFiles = environment["sizeFiles"]
len(sizeFiles)
totalBlocksWR = np.sum(sizeFiles)*environment["replicationFactor"]
print "Bloques: %s" %sizeFiles

cbl = np.linspace(-1,-1,totalBlocksWR).astype(int) #un array init. de -1
cbl_idf = np.linspace(-1,-1,totalBlocksWR).astype(int)
idc = 0
for idFile,blockF in enumerate(sizeFiles): #para cada file
#     print idFile,blockF
     for block in range(blockF*environment["replicationFactor"]): #para el numero total de bloques
#         print block
         if block%environment["replicationFactor"]==0: previousVMI=[]
         vmi = random.randint(0,len(setVM)-1)
#         print vmi
         tries = 200
#         print cbl_availability(vmi,cbl,idFile)
         while vmi in previousVMI  or not cbl_availability(vmi,setVM,cbl,cbl_idf,idFile):
             vmi = random.randint(0,len(setVM)-1)
#             print "."
             tries-=1
             if tries==0: 
                 print "ERROR - ASSIGNAMENT BLOCK TO VM"
                 print "Añadimos una nueva VM"
                 setVM = list(setVM)+list(np.random.choice(casesOfVM,1))
                 print "HERE"
                 tries = 200
 
#         print "\t",idc
         cbl[idc]= vmi 
         cbl_idf[idc] = idFile
         idc+=1
         previousVMI.append(vmi)
#         print cbl
#         print cbl_idf

print cbl
#print cbl_idf
print len(setVM)
if len(np.where(cbl==len(setVM)))>0: print "[ERROR] Desconocido cbl[j]= VMi>Len(setVM)"


vmID=0
""" Devuelve la demanda de una MVi según su uso """
def demand(vmID,cbl,cbl_idf):
    indVM = np.where(cbl==vmID)
    uVM = [0,0,0]
    for ap in indVM[0]:
        ut = environment["fileU"][cbl_idf[ap]]/environment["replicationFactor"]
        uVM = np.add(ut,uVM)
    return uVM
## TESTING Asignacion de VM a PM
def space_availabilityPM(pmi,cvm,vmID,setVM,cbl,cbl_idf):
    #La demanda de la VM
    #A travez de los bloques - Files - Demanda /3
    uVMnew = demand(vmID,cbl,cbl_idf)
    
    #Hemos de contabilizar la demanda que soporta la PMI actualmente, 
    #en su uso
    uPMcurrent = [0,0,0]
    for idx in range(len(cvm)):
        if cvm[idx]==pmi:
            uvm = demand(idx,cbl,cbl_idf)
            uPMcurrent = np.add(uPMcurrent,uvm)

    #La actual más la posible futura demanda    
    uPMfut = np.add(uPMcurrent,uVMnew)
    
     
    pmCore = environment["PM"][pmi]["Core"]
    pmHD = environment["PM"][pmi]["HD"]
    pmMM= environment["PM"][pmi]["MM"]

    pmThresCore = environment["PM"][pmi]["tresholdCPU"]
    pmThresHD = environment["PM"][pmi]["tresholdHD"]
    pmThresMM = environment["PM"][pmi]["tresholdMM"]
    

    if pmCore*pmThresCore < uPMfut[0]: return False
    if pmHD*pmThresHD < blockToTeras(uPMfut[2]): return False        
    if pmMM*pmThresMM < MbToGB(uPMfut[1]): return False        

    return True


totalVM = len(setVM)
totalPM = environment["numberOfPM"]
cvm = np.linspace(-1,-1,totalVM).astype(int)
for idx in range(totalVM):
    pmi = random.randint(0,totalPM-1)
    tries = 200
    while not space_availabilityPM(pmi,cvm,idx,setVM,cbl,cbl_idf):   
        pmi = random.randint(0,totalPM-1)
        tries -=1
        if tries==0: 
                 print "ERROR - NO hay maquinas" #ESTO NO DEBERIA DE PASAR
    cvm[idx]= pmi              

print cvm

#==============================================================================
#==============================================================================
# # GATT
#==============================================================================
#==============================================================================

gatt = g.GATT(environment,casesOfVM,seed=10)
sizePopulation = 30 # different VM 
gatt.population(sizePopulation)
#gatt.show()


#### Mutación 3. SWAP de MV

def getUtilizationPM(pmi,cvm,cbl,cbl_idf):
    uPMcurrent = [0,0,0]
    for idx in range(len(cvm)):
        if cvm[idx]==pmi:
            uvm = demand(idx,cbl,cbl_idf)
            uPMcurrent = np.add(uPMcurrent,uvm)
    return uPMcurrent
    
def getMaxUtilizationPM(pmi):
    pmCore = environment["PM"][pmi]["Core"]
    pmHD = environment["PM"][pmi]["HD"]
    pmMM= environment["PM"][pmi]["MM"]
    pmThresCore = environment["PM"][pmi]["tresholdCPU"]
    pmThresHD = environment["PM"][pmi]["tresholdHD"]
    pmThresMM = environment["PM"][pmi]["tresholdMM"]
    return pmCore*pmThresCore,pmMM*pmThresMM,pmHD*pmThresHD
  
vmIdo = 2
vmIdd = 4         
def is_SWAP_MV_suitable(vmIdo,vmIdd,setVM,cvm,cbl,cbl_idf):
    #La carga que soporta la máquina original es:
    uVMo = demand(vmIdo,cbl,cbl_idf)       
    uVMd = demand(vmIdd,cbl,cbl_idf)       
    #Se calcula las utilizaciones actuales de las diferentes PM
    idPMo = cvm[vmIdo]
    idPMd = cvm[vmIdd]
    uPMo = getUtilizationPM(idPMo,cvm,cbl,cbl_idf)
    uPMd = getUtilizationPM(idPMd,cvm,cbl,cbl_idf)
    #Se calculan los máximos de PM
    uPMto = getMaxUtilizationPM(idPMo)
    uPMtd = getMaxUtilizationPM(idPMd)

    #La nueva utilización_Origen = actual - origen + destino
    newUPMo = uPMo-uVMo+uVMd
    
    if uPMto[0] < newUPMo[0]: return False
    if uPMto[2] < blockToTeras(newUPMo[2]): return False        
    if uPMto[1] < MbToGB(newUPMo[1]): return False        

    newUPMd = uPMd-uVMd+uVMo

    if uPMtd[0] < newUPMd[0]: return False
    if uPMtd[2] < blockToTeras(newUPMd[2]): return False        
    if uPMtd[1] < MbToGB(newUPMd[1]): return False    
    
    return True


""" M3 - SWAP A VM """
cit = gatt.getCitizen(1)
#Dado un ciudadno se implementa un swap

setVM = cit["setVM"]
totalVM = len(setVM)
cvm = cit["cvm"]
cbl = cit["cbl"]
cbl_idf = cit["cbl_idf"]
#Se eligen origen y destino
orig = random.randint(0,totalVM-1) #La posición 22, tiene una máquina virtual del tipo: cvm[orig]["id"]
dest = random.randint(0,totalVM-1)
previousSelection = []
while (not setVM[dest]["id"] in previousSelection
       and setVM[dest]["id"] ==setVM[orig]["id"] 
       and is_SWAP_MV_suitable(orig,dest,setVM,cvm,cbl,cbl_idf)):
    dest = random.randint(0,totalVM-1)
    previousSelection.append(setVM[dest]["id"])
    if len(previousSelection)==len(casesOfVM):
        print "[Warning] No hay ninguna VM-PM que soporte: M3-swap"
        dest = -1
        break
if dest >=0: #Todo ha ido bien
    print "Swap OK: MV:%i  <-=-> MV:%i " %(orig,dest)
    #En cada se intercambia el ID de la VM
    cbl
    np.place(cbl,cbl==orig,-1)
    np.place(cbl,cbl==dest,-2)
    np.place(cbl,cbl==-1,dest)
    np.place(cbl,cbl==-2,orig)
    cbl
else:
    print "Swap Fallido"


""" M1 -create a new Virtual Machine  """

cit = gatt.getCitizen(1)
setVM = cit["setVM"]
cvm = cit["cvm"]
cbl = cit["cbl"]
cbl_idf = cit["cbl_idf"]

def cbl_availabilityNewVM(pmi,cbl,cbli,cvm,cbl_idf):
    fcbl = cbl
    fcbl[cbli]=len(cvm)
    fcvm = np.append(cvm,[pmi])
    uPMo = getUtilizationPM(pmi,fcvm,fcbl,cbl_idf)
    uPMto = getMaxUtilizationPM(pmi)
    if uPMto[0] < uPMo[0]: return False
    if uPMto[2] < blockToTeras(uPMo[2]): return False        
    if uPMto[1] < MbToGB(uPMo[1]): return False    
    return True
    
vmi = random.randint(0,len(casesOfVM)-1)
pmi = random.randint(0,totalPM-1)
cbli = random.randint(0,len(cbl)-1)
tries = 200
previousPMI = []
allOK=True
while ( pmi in previousPMI or not cbl_availabilityNewVM(pmi,cbl,cbli,cvm,cbl_idf)):
    previousPMI.append(pmi)
    pmi = random.randint(0,totalPM-1)
    tries-=1
    if tries==0 or  len(np.unique(previousPMI))==len(environment["PM"]): 
        allOK = False
        print "ERROR - M1 - No hay posibilidad de añadir una nueva VM a un PM"
if allOK: 
    print "OK"
    setVM.append(casesOfVM[vmi])
    cbl[cbli]=len(cvm)
    cvm = np.append(cvm,[pmi])

""" M2 -remove a Virtual Machine """ 
#Se elimina una VM si hay posibilidad de reasignar los bloques a otra nueva VM
#IN DATA
vmIdo = 2
vmIdd = 4         
def is_REMOVE_MV_suitable(vmIdo,vmIdd,setVM,cvm,cbl,cbl_idf):
    #La carga que soporta la máquina original es:
    uVMo = demand(vmIdo,cbl,cbl_idf)       
    #Se calcula las utilizaciones actuales de las diferentes PM
    idPMd = cvm[vmIdd]
    uPMd = getUtilizationPM(idPMd,cvm,cbl,cbl_idf)
    #Se calculan los máximos de PM
    uPMtd = getMaxUtilizationPM(idPMd)

    #La nueva utilización_Destino = actual + origen
    newUPMd = uPMd+uVMo

    if uPMtd[0] < newUPMd[0]: return False
    if uPMtd[2] < blockToTeras(newUPMd[2]): return False        
    if uPMtd[1] < MbToGB(newUPMd[1]): return False    
    
    return True

cit2 = gatt.getCitizen(23)

setVM = cit["setVM"]
cvm = cit["cvm"]
cbl = cit["cbl"]
cbl_idf = cit["cbl_idf"]

#Se eligen origen y destino
vmi = random.randint(0,len(setVM)-1)
vmd = random.randint(0,len(setVM)-1)

print "VMI: %i --->  %d " %(vmi,vmd)
previousSelection = []
while (setVM[vmd]["id"] in previousSelection
       or vmi == vmd
       or setVM[vmd]["id"] ==setVM[vmi]["id"] 
       or not is_REMOVE_MV_suitable(vmi,vmd,setVM,cvm,cbl,cbl_idf)):
    previousSelection.append(setVM[vmd]["id"])
    vmd = random.randint(0,totalVM-1)
    if len(previousSelection)==len(casesOfVM):
        print "[Warning] No hay ninguna VM-PM que soporte: M2-remove"
        vmd = -1
        break
if vmd >=0: #Todo ha ido bien
    print "REMOVE OK MV:%i  =-> MV:%i " %(vmi,vmd)
    #En cada se intercambia el ID de la VM
    print cbl
#    np.place(cbl,cbl==dest,orig)
    np.place(cbl,cbl==vmi,vmd)#Al destino se le asignan los bloques de Ori
    print cbl
#    print cvm
    cvm = np.delete(cvm,vmi)
#    print cvm
#    print len(cvm)
#    print len(setVM)
    del setVM[vmi]
#    print len(setVM)
else:
    print "Remove Fallido"

print len(setVM)
del setVM[0]
del setVM[0]
del setVM[0]
del setVM[0]
len(setVM)




""" M4 Move VM to another PM """
gatt = g.GATT(environment,casesOfVM,seed=10)
sizePopulation = 30 # different VM 
gatt.population(sizePopulation)

cit = gatt.getCitizen(1)
setVM = cit["setVM"]
cvm = cit["cvm"]
cbl = cit["cbl"]
cbl_idf = cit["cbl_idf"]

def is_CHANGE_MV_suitable(vmi,pmi,setVM,cvm,cbl,cbl_idf):
    #La carga que soporta la máquina original es:
    uVMo = demand(vmi,cbl,cbl_idf)       
    #Se calcula las utilizaciones actuales de las diferentes PM
    uPMd = getUtilizationPM(pmi,cvm,cbl,cbl_idf)
    #Se calculan los máximos de PM
    uPMtd = getMaxUtilizationPM(pmi)

    #La nueva utilización_Destino = actual + origen
    newUPMd = uPMd+uVMo

    if uPMtd[0] < newUPMd[0]: return False
    if uPMtd[2] < blockToTeras(newUPMd[2]): return False        
    if uPMtd[1] < MbToGB(newUPMd[1]): return False    
    
    return True

#Se eligen origen y destino
vmi = random.randint(0,len(setVM)-1)
pmi = random.randint(0,environment["numberOfPM"]-1)

print "VMI: %i --->  %d " %(vmi,pmi)
previousSelection = []
while (pmi in previousSelection #No se encontraba alla
       or pmi == cvm[vmi] #Ya estaba asignada esa VM a PM
       or not is_CHANGE_MV_suitable(vmi,pmi,setVM,cvm,cbl,cbl_idf)):
    previousSelection.append(pmi)
    pmi = random.randint(0,environment["numberOfPM"]-1)
    if len(np.unique(previousSelection))==environment["numberOfPM"]: #D1
        print "[Warning] M4. No hay ninguna PM que soporte: Vmi-Change"
        pmi = -1
        break

if pmi >=0:
    print pmi
    cvm[vmi] = pmi #Se asgina la CVM a PMI
else:
    print "Move Fallido"



""" M6 SWAP CBL to another CBL """
gatt = g.GATT(environment,casesOfVM,seed=10)
sizePopulation = 30 # different VM 
gatt.population(sizePopulation)

def getFRBlock(cbli,cbl):
    cbliRep = cbli / environment["replicationFactor"]
    vmCBLi = [] #Resto de VM dentro de ese mismo bloque
    for value in range(0,environment["replicationFactor"]):
        vmCBLi.append(cbl[cbliRep*environment["replicationFactor"]+value])    
    return vmCBLi
    
def is_SWAP_CBL_suitable(cbli,cblj,setVM,cvm,cbl,cbl_idf):
    
    vmi = cbl[cbli]  #Se asgina al bloque la nueva VM
    vmj = cbl[cblj]
    cblF = cbl
    cblF[cbli] = vmj
    cblF[cblj] = vmi
    
    #Se calcula la nueva demanda de cada VM
    uVMi = demand(vmi,cblF,cbl_idf)       
    uVMj = demand(vmj,cblF,cbl_idf)       

    #Se calcula la nueva demanda de cada PM
    pmi = cvm[vmi]
    pmj = cvm[vmj]
    uPMdi = getUtilizationPM(pmi,cvm,cblF,cbl_idf)
    uPMdj = getUtilizationPM(pmj,cvm,cblF,cbl_idf)
    
    #Se calculan los máximos de PM
    uPMti = getMaxUtilizationPM(pmi)
    uPMtj = getMaxUtilizationPM(pmj)
    
    newUPMi = uPMdi+uVMi

    if uPMti[0] < newUPMi[0]: return False
    if uPMti[2] < blockToTeras(newUPMi[2]): return False        
    if uPMti[1] < MbToGB(newUPMi[1]): return False    
    
    newUPMj = uPMdj+uVMj
    
    if uPMtj[0] < newUPMj[0]: return False
    if uPMtj[2] < blockToTeras(newUPMj[2]): return False        
    if uPMtj[1] < MbToGB(newUPMj[1]): return False    

    return True
    
cit = gatt.getCitizen(1)
setVM = cit["setVM"]
cvm = cit["cvm"]
cbl = cit["cbl"]
cbl_idf = cit["cbl_idf"]

#Puedo sustituir dos bloque sino tienen VM en común dentro del factor de replicado
cbli = random.randint(0,len(cbl)-1)
vmCBLi = getFRBlock(cbli,cbl) 
  
cblj = random.randint(0,len(cbl)-1)
vmCBLj = getFRBlock(cblj,cbl)

z = np.hstack((vmCBLi,vmCBLj))
tries = 200
cjj=-1
while cjj<0 and tries>0:
    print "CBLi: %i ----> CBLj: %i" %(cbli,cblj)
    while (not environment["replicationFactor"]*2==len(np.unique(z))):
        cblj = random.randint(0,len(cbl)-1)
        vmCBLj = getFRBlock(cblj,cbl)
        z = np.hstack((vmCBLi,vmCBLj))
        if tries==0:
            break
        tries-=1
    #Combinación de bloques factibles
    #Puedo cambiar ALGUNO de los tres bloques de vmBCLj
    if tries >0:
        for cblj in vmCBLj:
            if is_SWAP_CBL_suitable(cbli,cblj,setVM,cvm,cbl,cbl_idf):
                print "OK"
                cjj = cblj
                break
            
        if cjj<0:
            z=[]
            
if cjj>0:
    vmi = cbl[cbli]  #Se asgina al bloque la nueva VM
    cbl[cbli] = cbl[cjj]
    cbl[cjj] = vmi
else:
    print "Move Fallido"

""" M6 Move CBL to VM """
gatt = g.GATT(environment,casesOfVM,seed=10)
sizePopulation = 30 # different VM 
gatt.population(sizePopulation)

cit = gatt.getCitizen(1)
setVM = cit["setVM"]
cvm = cit["cvm"]
cbl = cit["cbl"]
cbl_idf = cit["cbl_idf"]


def is_CHANGE_CBL_suitable(cbli,vmi,setVM,cvm,cbl,cbl_idf):
    #La VM tendrá más carga
    #Esa carga la soporta su PMi
    
    cblF = cbl
    cblF[cbli] = vmi
    uVMo = demand(vmi,cblF,cbl_idf)   
    pmi = cvm[vmi]
    uPMd = getUtilizationPM(pmi,cvm,cblF,cbl_idf)
    #Se calculan los máximos de PM
    uPMtd = getMaxUtilizationPM(pmi)

    #La nueva utilización_Destino = actual + origen
    newUPMd = uPMd+uVMo

    if uPMtd[0] < newUPMd[0]: return False
    if uPMtd[2] < blockToTeras(newUPMd[2]): return False        
    if uPMtd[1] < MbToGB(newUPMd[1]): return False    
    
    return True



cbli = random.randint(0,len(cbl)-1)
#Control del factor de replicacion
cbliRep = cbli / environment["replicationFactor"]
VmReplicactionCBLValues = [] #Resto de VM dentro de ese mismo bloque
for value in range(0,environment["replicationFactor"]):
    VmReplicactionCBLValues.append(cbl[cbliRep*environment["replicationFactor"]+value])

#La nueva VM no ha de estar dentro de los valores de replicacion anteriores, ya se incluye la propia maquina
vmi = random.randint(0,len(setVM)-1)
print "CBLi: %i ---> VM: %d " %(cbli,vmi)
previousSelection = VmReplicactionCBLValues
while (vmi in previousSelection #No se encontraba alla
       or not is_CHANGE_CBL_suitable(cbli,vmi,setVM,cvm,cbl,cbl_idf)):
    previousSelection.append(vmi)
    vmi = random.randint(0,len(setVM)-1)
    if len(np.unique(previousSelection))==len(setVM): #D1
        print "[Warning] M6. No hay ninguna VM que soporte: CBLi-Change"
        vmi = -1
        break

if vmi >=0:
    print vmi
    cbl[cbli] = vmi #Se asgina al bloque la nueva VM
else:
    print "Move Fallido"

         
           
#==============================================================================
# # FITNESS TEST
#==============================================================================


gatt = g.GATT(environment,casesOfVM,seed=10)
sizePopulation = 30 # different VM 
gatt.population(sizePopulation)



#Data
cit = gatt.getCitizen(28)
setVM = cit["setVM"]
cvm = cit["cvm"]
cbl = cit["cbl"]
cbl_idf = cit["cbl_idf"]

#Potential cost of wasted resources
def getWasted_PM(pmi,cvm,cbl,cbl_idf):
    epsilon = 0.15
    uPMd = getUtilizationPM(pmi,cvm,cbl,cbl_idf)
    desv = np.std(uPMd)
    cores = environment["PM"][pmi]["Core"]
    uPMd[0] = uPMd[0]/cores
    if np.sum(uPMd)==0:
        return 0.0 
    else:
        return (desv+epsilon)/np.sum(uPMd) 

def getPower_PM(pmi,cvm,cbl,cbl_idf):
    uPMd = getUtilizationPM(pmi,cvm,cbl,cbl_idf)
#    print "CPU : U: %s" %uPMd[0]
    p_cpu = (environment["PM"][pmi]["CPU_full"]-environment["PM"][pmi]["CPU_idle"])*uPMd[0]+environment["PM"][pmi]["CPU_idle"]
    #Power memory
    s_mod = 0 #EN MB
    vmPMi = np.where(cvm==pmi)
    for vmi in vmPMi[0]:
        blokVMi = np.where(cbl==vmi)
        for blocki in blokVMi[0]:
            #Se los ficheros que usa
            s_mod +=environment["fileU"][cbl_idf[blocki]][1] #Valor de la memoria que consume ese fichero
        
    s_mod = MbToGB(s_mod)
    n_mod = 1 
    ro = 0.9
    u_MM = MbToGB(uPMd[1])/ environment["PM"][pmi]["MM"] #capacidad
    p_mm = n_mod*s_mod*ro*u_MM  
    #Power HD
#    print "HD: Teras Usados_HD: %s" %blockToTeras(uPMd[2])
    U_hd = blockToTeras(uPMd[2]) / environment["PM"][pmi]["HD"]
    p_hd = (1+0.4*U_hd)*environment["PM"][pmi]["HD_idle"]
    return p_cpu+p_mm+p_hd
    
    
def getFailure_System(pmi,cvm,cbl,cbl_idf,setVM):
    total_failure = 0.0
    for filei in range(len(environment["sizeFiles"])):
#        print filei
        #Bloques donde se usa ese fichero
        idxBloq = np.where(cbl_idf==filei)
        #MV donde cada bloque es asignado
        vm = cbl[idxBloq]
#        print vm
        #Para cada asignación de VM se suma el MTTF
        failure = 0.0
        for vmi in vm:
#            print vmi
#            print vmi
            failure += 1/setVM[vmi]["failure"]
            pmi = cvm[vmi]
            failure += 1/environment["PM"][pmi]["failure"]
        total_failure += failure
    return total_failure
    
fit = np.zeros((environment["numberOfPM"],3))    
for pmi in range(environment["numberOfPM"]):
    cores = environment["PM"][pmi]["Core"]
    print pmi,cores
    w= getWasted_PM(pmi,cvm,cbl,cbl_idf)
    if w==0.0:
        p = 0.0
    else:
        p = getPower_PM(pmi,cvm,cbl,cbl_idf)
    f = getFailure_System(pmi,cvm,cbl,cbl_idf,setVM)
    fit[pmi] =[w,p,f]
print fit
# Failure rate
type(fit)
#==============================================================================
# #FRENTE PARETO
#==============================================================================
def dominates(row, rowCandidate):
    return all(r >= rc for r, rc in zip(row, rowCandidate))

def cull(pts, dominates):
    dominated = []
    cleared = []
    remaining = pts
    while remaining:
        candidate = remaining[0]
        new_remaining = []
        for other in remaining[1:]:
            [new_remaining, dominated][dominates(candidate, other)].append(other)
        if not any(dominates(other, candidate) for other in new_remaining):
            cleared.append(candidate)
        else:
            dominated.append(candidate)
        remaining = new_remaining
    return cleared, dominated

## Computo frente pareto    
paretoPoints,dominatedPoints = cull(fit.tolist(),dominates)
print "*"*8 + " non-dominated answers " + ("*"*8)
for p in paretoPoints:
    print p
print "*"*8 + " dominated answers " + ("*"*8)
for p in dominatedPoints:
    print p
frentePareto = np.array(paretoPoints)    

Wmax = np.max(frentePareto[:,0])
Wmin = np.min(frentePareto[:,0]) 
Pmax = np.max(frentePareto[:,1])
Pmin = np.min(frentePareto[:,1]) 
Fmax = np.max(frentePareto[:,2])
Fmin = np.min(frentePareto[:,2]) 
   
MinW = np.sum(fit[:,0])
MinP = np.sum(fit[:,1])  
MinF = np.sum(fit[:,2]) 

fitValue = MinW/3*(Wmax-Wmin)
fitValue += MinP/3*(Pmax-Pmin)
fitValue += MinF/3*(Fmax-Fmin)

print fitValue
#gatt.show()
#


#==============================================================================
#==============================================================================
# # #EVOLUCION: 
#==============================================================================
#==============================================================================

def C1_OneCuttingPoint(citOri,citDest):
    citM = gatt.getCitizen(citOri)
    citF = gatt.getCitizen(citDest)
    cvmM = citM["cvm"]
    cblM = citM["cbl"]
    cvmF = citF["cvm"]
    cblF = citF["cbl"]
    cbl_idfM = citM["cbl_idf"]
    cbl_idfF = citF["cbl_idf"]
    ##ID1 One cutting-point crossover 
    i = random.randint(0,np.min((len(cvmM),len(cvmF)))-1)
    j = random.randint(0,(len(cblM)-1)/3)*3 # Valor J ha de ser %ReplifcationFactor == 0 para asegurar el grado de replicas
    #Hijos            
    h1_cvm = np.hstack((cvmM[i:],cvmF[:i]))
    h2_cvm = np.hstack((cvmM[:i],cvmF[i:]))
    h1_cbl = np.hstack((cblM[j:],cblF[:j]))
    h2_cbl = np.hstack((cblM[:j],cblF[j:]))
    h1_cbl_idf = np.hstack((cbl_idfM[j:],cbl_idfF[:j]))
    h2_cbl_idf = np.hstack((cbl_idfM[:j],cbl_idfF[j:]))
    h1_setVM = np.hstack((citM["setVM"][i:],citF["setVM"][:i]))
    h2_setVM = np.hstack((citM["setVM"][:i],citF["setVM"][i:]))
    h1_setVM = h1_setVM.tolist()
    h2_setVM = h2_setVM.tolist()
    return h1_cvm,h1_cbl,h1_cbl_idf,h1_setVM,h2_cvm,h2_cbl,h2_cbl_idf,h2_setVM


def C2_TwoCuttingPoint(citOri,citDest):
    citM = gatt.getCitizen(citOri)
    citF = gatt.getCitizen(citDest)
    cvmM = citM["cvm"]
    cblM = citM["cbl"]
    cvmF = citF["cvm"]
    cblF = citF["cbl"]
    cbl_idfM = citM["cbl_idf"]
    cbl_idfF = citF["cbl_idf"]
    i0 = random.randint(0,np.min((len(cvmM),len(cvmF)))-1)
    i1 = random.randint(i0,np.min((len(cvmM),len(cvmF)))-1)
    j0 = random.randint(0,(len(cblM)-1)/3)*3
    j1 = random.randint(j0/3,(len(cblM)-1)/3)*3# Valor J ha de ser %ReplifcationFactor == 0 para asegurar el grado de replicas
     
    #Hijos            
    h1_cvm = np.hstack((cvmM[:i0],cvmF[i0:i1],cvmM[i1:]))
    h2_cvm = np.hstack((cvmF[:i0],cvmM[i0:i1],cvmF[i1:]))
    h1_cbl = np.hstack((cblM[:j0],cblF[j0:j1],cblM[j1:]))
    h2_cbl = np.hstack((cblF[:j0],cblM[j0:j1],cblF[j1:]))
    
    h1_setVM = np.hstack((citM["setVM"][:i0],citF["setVM"][i0:i1],citM["setVM"][i1:]))
    h2_setVM = np.hstack((citF["setVM"][:i0],citM["setVM"][i0:i1],citF["setVM"][i1:]))
    h1_setVM = h1_setVM.tolist()
    h2_setVM = h2_setVM.tolist()
    
    h1_cbl_idf = np.hstack((cbl_idfM[:j0],cbl_idfF[j0:j1],cbl_idfM[j1:]))
    h2_cbl_idf = np.hstack((cbl_idfF[:j0],cbl_idfM[j0:j1],cbl_idfF[j1:]))
                
    
    return h1_cvm,h1_cbl,h1_cbl_idf,h1_setVM,h2_cvm,h2_cbl,h2_cbl_idf,h2_setVM
    
def C3_NonUniform(citOri,citDest):
    citM = gatt.getCitizen(citOri)
    citF = gatt.getCitizen(citDest)
    cvmM = citM["cvm"]
    cblM = citM["cbl"]
    cvmF = citF["cvm"]
    cblF = citF["cbl"]
    cbl_idfM = citM["cbl_idf"]
    cbl_idfF = citF["cbl_idf"]
    ##ID1 One cutting-point crossover 
    i0 = random.randint(0,np.min(len(cvmM))-1)
    i1 = random.randint(0,np.min(len(cvmF))-1)
    
    j0 = random.randint(0,(len(cblM)-1)/3)*3
    j1 = random.randint(0,(len(cblF)-1)/3)*3
    #Hijos            
    h1_cvm = np.hstack((cvmM[:i0],cvmF[i1:]))
    h2_cvm = np.hstack((cvmF[:i1],cvmM[i0:]))
    
    h1_cbl = np.hstack((cblM[:j0],cblF[j1:]))
    h2_cbl = np.hstack((cblF[:j1],cblM[:j0]))
    
    h1_cbl_idf = np.hstack((cbl_idfM[:j0],cbl_idfF[j1:]))
    h2_cbl_idf = np.hstack((cbl_idfF[:j1],cbl_idfM[:j0]))
    
    h1_setVM = np.hstack((citM["setVM"][:i0],citF["setVM"][i1:]))
    h2_setVM = np.hstack((citF["setVM"][:i1],citM["setVM"][i0:]))
    h1_setVM = h1_setVM.tolist()
    h2_setVM = h2_setVM.tolist()
    return h1_cvm,h1_cbl,h1_cbl_idf,h1_setVM,h2_cvm,h2_cbl,h2_cbl_idf,h2_setVM
    
#Suponemos fitness
#fit = np.random.rand(20)
fit  = gatt.fit
crossOverFunction = C1_OneCuttingPoint
citizenid = 0
crossover_treshold = 0.9

candidates = np.sort(fit)
p_candidates = []
p_cand = []
for value in candidates: p_candidates.append(1/(value/np.sum(candidates)))
for value in p_candidates: p_cand.append(value/np.sum(p_candidates))
    
#Creación de una nueva población
allOk = 10000
previousNoFactibleFathers = []
while citizenid<sizePopulation and allOk>=0:
    #==============================================================================
    # #Seleccion: Roulette-wheel selection
    #==============================================================================
    #Los mejores están mas cerca del cero
    candys = np.random.choice(candidates,2,replace=False,p=p_cand)
    citOri = np.where(fit==candys[0])[0][0]
    citDest = np.where(fit==candys[1])[0][0]
    if not [citOri,citDest] in previousNoFactibleFathers:
        print "Crossing parents: %i - %i" %(citOri,citDest)
        #Applying one crossover

        if random.random()<crossover_treshold:
            #==============================================================================
            # CrossOverS
            #==============================================================================
            future_population = {}
            citM = gatt.getCitizen(citOri)
            citF = gatt.getCitizen(citDest)
            h1_cvm,h1_cbl,h1_cbl_idf,h1_setVM,h2_cvm,h2_cbl,h2_cbl_idf,h2_setVM = crossOverFunction(citOri,citDest)
    
            possible = True
            for pmi in range(environment["numberOfPM"]):
                #TESTING
                print getFailure_System(pmi,h1_cvm,h1_cbl,h1_cbl_idf,h1_setVM)
                print getFailure_System(pmi,h2_cvm,h2_cbl,h2_cbl_idf,h2_setVM)                
                #END TESTING                
                upmih1 = getUtilizationPM(pmi,h1_cvm,h1_cbl,h1_cbl_idf)
                upmih2 = getUtilizationPM(pmi,h2_cvm,h2_cbl,h2_cbl_idf)
                uMax = getMaxUtilizationPM(pmi)
                if uMax[0] < upmih1[0]: possible= False
                if uMax[2] < blockToTeras(upmih1[2]): possible = False        
                if uMax[1] < MbToGB(upmih1[1]): possible = False    
                if uMax[0] < upmih2[0]: possible= False
                if uMax[2] < blockToTeras(upmih2[2]): possible = False        
                if uMax[1] < MbToGB(upmih2[1]): possible = False   
                if not possible: break
            
            if possible:
                print "Creating two new citizens"
                
                fit_value = 100 #TEMPORAL
                fitness_values = np.random.rand(environment["numberOfPM"]) #TEMPORAL
                
            #    fitness_values = self.get_fitness_citizen(h1_cvm,h1_cbl,cbl_idf,h1_setVM])
            #    fit_value = self.get_Fitness_value(fitness_values)
                future_population[citizenid] = {"cvm":h1_cvm,"cbl":h1_cbl,"setVM":h1_setVM,
                                            "cbl_idf":h1_cbl_idf, "fit_value":fit_value,"fit_values":fitness_values}
            
                citizenid+=1
            #    fitness_values = self.get_fitness_citizen(h2_cvm,h2_cbl,cbl_idf,h2_setVM)
            #    fit_value = self.get_Fitness_value(fitness_values)
                future_population[citizenid] = {"cvm":h2_cvm,"cbl":h2_cbl,"setVM":h2_setVM,
                                            "cbl_idf":h2_cbl_idf,"fit_value":fit_value,"fit_values":fitness_values}
                citizenid+=1  
            else:
                previousNoFactibleFathers.append([citOri,citDest])                         
        else: #No crossover
            if not np.any(future_population.values() == citM) and not np.any(future_population.values() == citF):
                print "Populating the next generation with fathers"
                future_population[citizenid] = citM
                citizenid+=1
                future_population[citizenid] = citF
                citizenid+=1 
     #Padres anteriormente contrastados           
    else: 
        allOk-=1 #Evitar bucle infinito
        
#print future_population
print len(future_population)     
print allOk
            
            
#==============================================================================
#==============================================================================
# # #EVOLUCION: TwoCuttingPoint  TEST TEST TEST TEST 
#==============================================================================
#==============================================================================
#Suponemos fitness
#fit = np.random.rand(20)
fit  = gatt.fit

citizenid = 0
crossover_treshold = 0.9

candidates = np.sort(fit)
p_candidates = []
p_cand = []
for value in candidates: p_candidates.append(1/(value/np.sum(candidates)))
for value in p_candidates: p_cand.append(value/np.sum(p_candidates))
    
#Creación de una nueva población
allOk = 10000
previousNoFactibleFathers = []
while citizenid<sizePopulation and allOk>=0:
    #==============================================================================
    # #Seleccion: Roulette-wheel selection
    #==============================================================================
    #Los mejores están mas cerca del cero
    candys = np.random.choice(candidates,2,replace=False,p=p_cand)
    citOri = np.where(fit==candys[0])[0][0]
    citDest = np.where(fit==candys[1])[0][0]
    if not [citOri,citDest] in previousNoFactibleFathers:
        print "Crossing parents: %i - %i" %(citOri,citDest)
        #Applying one crossover

        if random.random()<crossover_treshold:
            #==============================================================================
            # CrossOverS
            #==============================================================================
            future_population = {}
            citM = gatt.getCitizen(citOri)
            citF = gatt.getCitizen(citDest)
            cvmM = citM["cvm"]
            cblM = citM["cbl"]
            cvmF = citF["cvm"]
            cblF = citF["cbl"]
            cbl_idfM = citM["cbl_idf"]
            cbl_idfF = citF["cbl_idf"]
            ##ID1 One cutting-point crossover 
            i0 = random.randint(0,np.min((len(cvmM),len(cvmF)))-1)
            i1 = random.randint(i0,np.min((len(cvmM),len(cvmF)))-1)
            j0 = random.randint(0,(len(cblM)-1)/3)*3
            j1 = random.randint(j0/3,(len(cblM)-1)/3)*3# Valor J ha de ser %ReplifcationFactor == 0 para asegurar el grado de replicas
         
            #Hijos            
            h1_cvm = np.hstack((cvmM[:i0],cvmF[i0:i1],cvmM[i1:]))
            h2_cvm = np.hstack((cvmF[:i0],cvmM[i0:i1],cvmF[i1:]))
            h1_cbl = np.hstack((cblM[:j0],cblF[j0:j1],cblM[j1:]))
            h2_cbl = np.hstack((cblF[:j0],cblM[j0:j1],cblF[j1:]))
    
            possible = True
            for pmi in range(environment["numberOfPM"]):
                upmih1 = getUtilizationPM(pmi,h1_cvm,h1_cbl,cbl_idf)
                upmih2 = getUtilizationPM(pmi,h2_cvm,h2_cbl,cbl_idf)
                uMax = getMaxUtilizationPM(pmi)
                if uMax[0] < upmih1[0]: possible= False
                if uMax[2] < blockToTeras(upmih1[2]): possible = False        
                if uMax[1] < MbToGB(upmih1[1]): possible = False    
                if uMax[0] < upmih2[0]: possible= False
                if uMax[2] < blockToTeras(upmih2[2]): possible = False        
                if uMax[1] < MbToGB(upmih2[1]): possible = False   
                if not possible: break
            
            if possible:
                print "Creating two new citizens"

                
                
                fit_value = 100 #TEMPORAL
                fitness_values = np.random.rand(environment["numberOfPM"]) #TEMPORAL
                
            #    fitness_values = self.get_fitness_citizen(h1_cvm,h1_cbl,cbl_idf,h1_setVM])
            #    fit_value = self.get_Fitness_value(fitness_values)
                future_population[citizenid] = {"cvm":h1_cvm,"cbl":h1_cbl,"setVM":h1_setVM,
                                            "cbl_idf":h1_cbl_idfM, "fit_value":fit_value,"fit_values":fitness_values}
            
                citizenid+=1
            #    fitness_values = self.get_fitness_citizen(h2_cvm,h2_cbl,cbl_idf,h2_setVM)
            #    fit_value = self.get_Fitness_value(fitness_values)
                future_population[citizenid] = {"cvm":h2_cvm,"cbl":h2_cbl,"setVM":h2_setVM,
                                            "cbl_idf":h2_cbl_idfM,"fit_value":fit_value,"fit_values":fitness_values}
                citizenid+=1  
            else:
                previousNoFactibleFathers.append([citOri,citDest])                         
        else: #No crossover
            if not np.any(future_population.values() == citM) and not np.any(future_population.values() == citF):
                print "Populating the next generation with fathers"
                future_population[citizenid] = citM
                citizenid+=1
                future_population[citizenid] = citF
                citizenid+=1 
     #Padres anteriormente contrastados           
    else: 
        allOk-=1 #Evitar bucle infinito
        
#print future_population
print len(future_population)     
print allOk



#==============================================================================
#==============================================================================
# # #EVOLUCION: Non-uniform #TEST TEST TEST TEST
#==============================================================================
#==============================================================================
#Suponemos fitness
#fit = np.random.rand(20)
fit  = gatt.fit

citizenid = 0
crossover_treshold = 0.9

candidates = np.sort(fit)
p_candidates = []
p_cand = []
for value in candidates: p_candidates.append(1/(value/np.sum(candidates)))
for value in p_candidates: p_cand.append(value/np.sum(p_candidates))
    
#Creación de una nueva población
allOk = 10000
previousNoFactibleFathers = []
while citizenid<sizePopulation and allOk>=0:
    #==============================================================================
    # #Seleccion: Roulette-wheel selection
    #==============================================================================
    #Los mejores están mas cerca del cero
    candys = np.random.choice(candidates,2,replace=False,p=p_cand)
    citOri = np.where(fit==candys[0])[0][0]
    citDest = np.where(fit==candys[1])[0][0]
    if not [citOri,citDest] in previousNoFactibleFathers:
        print "Crossing parents: %i - %i" %(citOri,citDest)
        #Applying one crossover

        if random.random()<crossover_treshold:
            #==============================================================================
            # CrossOverS
            #==============================================================================
            future_population = {}
            citM = gatt.getCitizen(citOri)
            citF = gatt.getCitizen(citDest)
            cvmM = citM["cvm"]
            cblM = citM["cbl"]
            cvmF = citF["cvm"]
            cblF = citF["cbl"]
            cbl_idfM = citM["cbl_idf"]
            cbl_idfF = citF["cbl_idf"]
            ##ID1 One cutting-point crossover 
            i0 = random.randint(0,np.min(len(cvmM))-1)
            i1 = random.randint(0,np.min(len(cvmF))-1)

            j0 = random.randint(0,(len(cblM)-1)/3)*3
            j1 = random.randint(0,(len(cblF)-1)/3)*3
#            j1 = random.randint(j0/3,(len(cblM)-1)/3)*3# Valor J ha de ser %ReplifcationFactor == 0 para asegurar el grado de replicas
         
            #Hijos            
            h1_cvm = np.hstack((cvmM[:i0],cvmF[i1:]))
            h2_cvm = np.hstack((cvmF[:i1],cvmM[i0:]))
            
            h1_cbl = np.hstack((cblM[:j0],cblF[j1:]))
            h2_cbl = np.hstack((cblF[:j1],cblM[:j0]))
            
            h1_cbl_idf = np.hstack((cbl_idfM[:j0],cbl_idfF[j1:]))
            h2_cbl_idf = np.hstack((cbl_idfF[:j1],cbl_idfM[:j0]))
            
            h1_setVM = np.hstack((citM["setVM"][:i0],citF["setVM"][i1:]))
            h2_setVM = np.hstack((citF["setVM"][:i1],citM["setVM"][i0:]))
            h1_setVM = h1_setVM.tolist()
            h2_setVM = h2_setVM.tolist()
    
            possible = True
            for pmi in range(environment["numberOfPM"]):
                upmih1 = getUtilizationPM(pmi,h1_cvm,h1_cbl,h1_cbl_idf)
                upmih2 = getUtilizationPM(pmi,h2_cvm,h2_cbl,h2_cbl_idf)
                uMax = getMaxUtilizationPM(pmi)
                if uMax[0] < upmih1[0]: possible= False
                if uMax[2] < blockToTeras(upmih1[2]): possible = False        
                if uMax[1] < MbToGB(upmih1[1]): possible = False    
                if uMax[0] < upmih2[0]: possible= False
                if uMax[2] < blockToTeras(upmih2[2]): possible = False        
                if uMax[1] < MbToGB(upmih2[1]): possible = False   
                if not possible: break
            
            if possible:
                print "Creating two new citizens"
               
                fit_value = 100 #TEMPORAL
                fitness_values = np.random.rand(environment["numberOfPM"]) #TEMPORAL
                
            #    fitness_values = self.get_fitness_citizen(h1_cvm,h1_cbl,cbl_idf,h1_setVM])
            #    fit_value = self.get_Fitness_value(fitness_values)
                future_population[citizenid] = {"cvm":h1_cvm,"cbl":h1_cbl,"setVM":h1_setVM,
                                            "cbl_idf":h1_cbl_idf, "fit_value":fit_value,"fit_values":fitness_values}
            
                citizenid+=1
            #    fitness_values = self.get_fitness_citizen(h2_cvm,h2_cbl,cbl_idf,h2_setVM)
            #    fit_value = self.get_Fitness_value(fitness_values)
                future_population[citizenid] = {"cvm":h2_cvm,"cbl":h2_cbl,"setVM":h2_setVM,
                                            "cbl_idf":h2_cbl_idf,"fit_value":fit_value,"fit_values":fitness_values}
                citizenid+=1  
            else:
                previousNoFactibleFathers.append([citOri,citDest])                         
        else: #No crossover
            if not np.any(future_population.values() == citM) and not np.any(future_population.values() == citF):
                print "Populating the next generation with fathers"
                future_population[citizenid] = citM
                citizenid+=1
                future_population[citizenid] = citF
                citizenid+=1 
     #Padres anteriormente contrastados           
    else: 
        allOk-=1 #Evitar bucle infinito
        
#print future_population
print len(future_population)     
print allOk








""" MUTATION PROCEDIMENT TESTING """

probabilityHappen = 0.4
def function1():
    return 1
def function2():
    return 2
def function3():
    return 3
def function4():
    return 4
    
mutations = [function1,function2,function3,function4]

if probabilityHappen :
    citi = random.randint(0,sizePopulation-1)
    cit = gatt.getCitizen(citi)
    muti = random.randint(0,len(mutations)-1)
    print mutations[muti]()