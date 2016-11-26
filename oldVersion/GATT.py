# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:07:09 2016

@author: isaac
"""

#import logging
import numpy as np
import random 
#logging.basicConfig(filename='gatt.log',level=logging.DEBUG)
#logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
"""
 - Includes GA operations
 - Defines: Citizen - fitness,creation, and mutations
"""
class GATT:
    
    __defaultTries = 200
    
    __chromosomeVM = ["cvm"]
    __chromosomeBL = ["cbl"]
    
    """ MUTATIONS on any chromosome  """
    """ M1 -create a new Virtual Machine / new BLOCK """
    def __M1_createCRH(self,citizenID,chromosome):
#        TODO: Elegir el id de la maquina 
        self.pop[citizenID][chromosome].append("x")
        
    """ M2 -remove a Virtual Machine / new BLOCK"""        
    def __M2_removeCRH(self,citizenID,chromosome):
        cid = random.randint(0,len(self.pop[citizenID][chromosome]))
#        self.pop[citizenID][chromosome] = np.delete(self.pop[citizenID][chromosome],cid)    
        #TODO: Restructurar asignación población
    """ M3 -swap a Virtual Machine """   
    def __M3_swapVM(self,citizenID):
#        totalVM = len(self.pop[citizendID]["setVM"])
#        cvm = self.pop[citizendID]["setVM"]
#        orig = random.randint(0,totalVM-1)
#        vm = np.random.choice(self.casesOfVM,1)
#        if vm["id"] = 
        return 3
        
    """ M4 -change value Virtual Machine """   
    def __M4_changeVM(self):
        return 3
       
  
    """ Operations """
    def __init__(self,cloudEnvironment,casesOfVM,crossover,seed):
        random.seed(seed)
        self.crossover = np.vectorize(crossover)
        self.pop = {}
        self.mutations = [self.__M1_createCRH,self.__M2_removeCRH,self.__M3_swapVM,self.__M4_changeVM]
        self.ce = cloudEnvironment
        self.casesOfVM = casesOfVM
        
        files = cloudEnvironment["JOBS"][:,1].astype(int)
        sizeFiles = cloudEnvironment["sizeFile"][files]
        capacityPM = self.__capacity(cloudEnvironment["PM"])
        totalBlocksWR = np.sum(sizeFiles)*cloudEnvironment["replicationFactor"]
        if totalBlocksWR>=self.__terasToBlock(capacityPM):
            print "Error"
        else:
            print "Creating is factible"
#        logging.error('Blocks over system capacity')            

    """ Teras to Block """
    def __terasToBlock(self,teras):
        return (teras*1024*1024)/self.ce["sizeBlock"]

    """ MB to Teras """
    def __MbToTera(mb):
        return mb/1024/1024        
        
    """ Obtain Tera capacity of set VM or PM """
    def __capacity(self,setM):
        totalCapacity = 0
        for ipm in setM:
            totalCapacity += ipm["hd"] 
        return totalCapacity
    
    def __space_availability(self,vmi,terasVMI,cbl):
        totalBlockCapacity = self.__terasToBlock(terasVMI)
        blockAssig = 0
        for idx in range(len(cbl)):
            if cbl[idx]==vmi:
                blockAssig+=1
        return totalBlockCapacity>blockAssig
        
        
    def __space_availabilityPM(self,pmi,cvm,vmi,setVM):
        totalCoreCapacity = self.ce["PM"][pmi]["cores"]
        totalHDCapacity = self.ce["PM"][pmi]["hd"]
        vmiCores= 0
        vmiHD = 0
        for idx in range(len(cvm)):
            if cvm[idx]==pmi:
                vmiCores +=setVM[vmi]["cores"]
                vmiHD += setVM[vmi]["hd"]
        return (totalCoreCapacity>=vmiCores and totalHDCapacity>=vmiHD)
  
    
    """ Get the minimun set of VM"""
    def __getMinSetVM(self):
        totalBlocks = np.sum(self.ce["sizeFile"])*self.ce["replicationFactor"]
        totalTeraBlocks = (totalBlocks*self.ce["sizeBlock"])/1024/1024
        
        numberVM = self.ce["replicationFactor"]
        setVM = np.random.choice(self.casesOfVM,numberVM)
        while self.__capacity(setVM)<totalTeraBlocks: #TODO factor ? .5
            numberVM +=1
            setVM = np.random.choice(self.casesOfVM,numberVM)
        return setVM
        
    def getCitizen(self,i):
        return self.pop[i]
    """ Creating initial Citizens-Chromosomes """
    # size - of population
    def population(self,size):
#        logging.info('Population')
        self.population = size
        files = self.ce["JOBS"][:,1].astype(int)
        sizeFiles = self.ce["sizeFile"][files]
        totalBlocksWR = np.sum(sizeFiles)*self.ce["replicationFactor"]
        
        setVM = self.__getMinSetVM()

        print "Número de máquinas virtuales min: %i" %len(setVM)
        #create citizens        
        for idCiti in range(size):
            print "Creando citizen: %i" %idCiti
            
            #Asignación de bloques a MV
            #TODO CONSIDERAR LA UTILIZACION
            print "\tAsignación de BLQ a MV"
            cbl = np.linspace(-1,-1,totalBlocksWR).astype(int) #un array init. de -1
            idc = 0
            for blockF in sizeFiles: #para cada file
                for block in range(blockF*self.ce["replicationFactor"]): #para el numero total de bloques
                    if block%self.ce["replicationFactor"]==0: previousVMI=[]
                    vmi = random.randint(0,len(setVM)-1)
                    tries = self.__defaultTries
                    terasVMI = setVM[vmi]["hd"]
                    while vmi in previousVMI  or not self.__space_availability(vmi,terasVMI,cbl):
                        vmi = random.randint(0,len(setVM)-1)
                        terasVMI = setVM[vmi]["hd"]
                        tries-=1
                        if tries==0: 
                            print "ERROR - ASSIGNAMENT BLOCK TO VM"
                            print "Añadimos una nueva VM"
                            setVM = list(setVM)+list(np.random.choice(self.casesOfVM,1))
                            tries = 200
                    cbl[idc]= vmi
                    idc+=1
                    previousVMI.append(vmi)

            #Asignación de MV a PM
            print "\tAsignación de MV a MP"
            totalVM = len(setVM)
            totalPM = self.ce["numberOfPM"]
            cvm = np.linspace(-1,-1,totalVM).astype(int)
            for idx in range(totalVM):
                pmi = random.randint(0,totalPM-1)
                tries = self.__defaultTries
                while not self.__space_availabilityPM(pmi,cvm,idx,setVM):
                    pmi = random.randint(0,totalPM-1)
                    tries -=1
                    if tries==0: 
                        print "ERROR - NO hay maquinas" #ESTO NO DEBERIA DE PASAR
                cvm[idx]= pmi   

            #Un ciudadano tiene un CVM, CBL, SETVM            
            self.pop[idCiti]={"cvm":cvm,"cbl":cbl,"setVM":setVM}
            
            
    """ Evolve operation """
    def evolve(self,seed):
        random.seed(seed)
        cit = random.randint(1,self.population)
        self.pop[cit]["cbl"] = self.crossover(self.pop[cit]["cbl"])
        
    """ Fitness operation """    
    def fitness(self):
        print "Fitness"
        
    """ Print Population """
    def show(self):
        for idx in self.pop.keys():
            print "Citizen: %i" %idx
            print "\tsetVM:\t%s" %self.pop[idx]["setVM"]
            print "\tCBL:\t%s" %self.pop[idx]["cbl"]
            print "\tCVM:\t%s" %self.pop[idx]["cvm"]