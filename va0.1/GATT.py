# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:07:09 2016

@author: isaac
"""

import logging
import numpy as np
import random 


#logging.basicConfig(filename='gatt.log',level=logging.DEBUG)
#logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
#
#logger = logging.getLogger('simpleExample')
#logger.debug('debug message')
"""
 - Includes GA operations
 - Defines: Citizen - fitness,creation, and mutations
"""
class GATT:
    


    __defaultTries = 200

   
   
    """ Operations """
    def __init__(self,environment,casesOfVM,seed):
        random.seed(seed)
        self.pop = {}
        self.ce = environment
        self.casesOfVM = casesOfVM
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # create a file handler
        handler = logging.FileHandler('gatt.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("Hello baby: I'm crazy")     
            
    def terasToBlock(self,teras):
        return (teras*1024*1024)/self.ce["sizeBlock"]
    def blockToTeras(self,block):
        return (block*self.ce["sizeBlock"])/1024/1024
    def MbToTera(self,mb):
        return mb/1024/1024 
    def MbToGB(self,mb):
        return mb/1024
    def getTotalCPU(self,):
        return np.sum(self.ce["fileU"][:,0])
    def getTotalMM(self,):
        return self.MbToGB(np.sum(self.ce["fileU"][:,1]))
    def getTotalHD(self,):
        return np.sum(self.ce["fileU"][:,2])   
        
    def __getDemandJobs(self):
        sizeFiles = self.ce["sizeFiles"]
        totalBlocksWR = np.sum(sizeFiles)*self.ce["replicationFactor"]
        
        totalTeras = self.blockToTeras(totalBlocksWR)
        
        uCPU_Cores = self.getTotalCPU()
        uMM_GB = self.getTotalMM()
        return uCPU_Cores,uMM_GB,totalTeras
        
    def __getMaxSetVM(self,setVM):
       nCores = 0
       GigasMM = 0
       TerasHD = 0
       for ipm in setVM:
            nCores += ipm["Core"]
            GigasMM += ipm["MM"]
            TerasHD += ipm["HD"] 
       return nCores,GigasMM,TerasHD
       
    def __getMaxAVM(self,aVM):
       nCores = aVM["Core"]
       GigasMM = aVM["MM"]
       TerasHD = aVM["HD"] 
       return nCores,GigasMM,TerasHD
    
    def __isSuitable(self,setVM):
       nCores,GigasMM,TerasHD= self.__getMaxSetVM(setVM)
       uCPU_Cores,uMM_GB,totalTeras = self.__getDemandJobs()
       if nCores<uCPU_Cores or GigasMM < uMM_GB or TerasHD < totalTeras:  return False
       return True    
       
        
    def __getMinSetVM(self):   
       numberVM = self.ce["replicationFactor"]
       setVM = np.random.choice(self.casesOfVM,numberVM)
       while not self.__isSuitable(setVM):
           numberVM +=1
           setVM = np.random.choice(self.casesOfVM,numberVM)
       return setVM
        
    
    def __demand(self,vmID,cbl,cbl_idf):
        indVM = np.where(cbl==vmID)
        uVM = [0,0,0]
        for ap in indVM[0]:
            ut = self.ce["fileU"][cbl_idf[ap]]/self.ce["replicationFactor"]
            uVM = np.add(ut,uVM)
        return uVM
        
    def __getUtilizationPM(self,pmi,cvm,cbl,cbl_idf):
        uPMcurrent = [0,0,0]
        for idx in range(len(cvm)):
            if cvm[idx]==pmi:
                uvm = self.__demand(idx,cbl,cbl_idf)
                uPMcurrent = np.add(uPMcurrent,uvm)
        return uPMcurrent
        
    def __getMaxUtilizationPM(self,pmi):
        pmCore = self.ce["PM"][pmi]["Core"]
        pmHD = self.ce["PM"][pmi]["HD"]
        pmMM= self.ce["PM"][pmi]["MM"]
        pmThresCore = self.ce["PM"][pmi]["tresholdCPU"]
        pmThresHD = self.ce["PM"][pmi]["tresholdHD"]
        pmThresMM = self.ce["PM"][pmi]["tresholdMM"]
        return pmCore*pmThresCore,pmMM*pmThresMM,pmHD*pmThresHD
        
        
    def __cbl_availability(self,idVM,setVM,cbl,cbl_idf,idFile,cvm_u):
        #La capacidad de la máquina
        nCores,GigasMM,TerasHD = self.__getMaxAVM(setVM[idVM])
        #La tarea influye 1/3 en el peso de la máquina

        uVM = cvm_u[idVM]
        uVM = np.add(self.ce["fileU"][idFile] / self.ce["replicationFactor"],uVM)

        #La nueva Uassginada es factible en esa VM?
        if uVM[0]>nCores:
#            print "Core"
            return False
        if self.MbToGB(uVM[1])>GigasMM:
#            print "MM"
            return False   
        if self.blockToTeras(uVM[2])>TerasHD:
#            print "HD"
            return False   
        return True 


    def __space_availabilityPM(self,pmi,cvm,vmID,setVM,cbl,cbl_idf):
        #La demanda de la VM
        self.logger.info("\t\t Space disponible en PM: %i" %pmi)
        #A travez de los bloques - Files - Demanda /3
        uVMnew = self.__demand(vmID,cbl,cbl_idf)
        #Hemos de contabilizar la demanda que soporta la PMI actualmente, 
        #en su uso
        uPMcurrent = self.__getUtilizationPM(pmi,cvm,cbl,cbl_idf)
        uPMfut = np.add(uPMcurrent,uVMnew)
         
        pmCore,pmHD,pmMM = self.__getMaxUtilizationPM(pmi)
    
        if pmCore < uPMfut[0]: return False
        if pmHD < self.blockToTeras(uPMfut[2]): return False        
        if pmMM < self.MbToGB(uPMfut[1]): return False        
    
        return True

      
    """ Para M3 """
    def __is_SWAP_MV_suitable(self,vmIdo,vmIdd,setVM,cvm,cbl,cbl_idf):
        #La carga que soporta la máquina original es:
        uVMo = self.__demand(vmIdo,cbl,cbl_idf)       
        uVMd = self.__demand(vmIdd,cbl,cbl_idf)       
        #Se calcula las utilizaciones actuales de las diferentes PM
        idPMo = cvm[vmIdo]
        idPMd = cvm[vmIdd]
        uPMo = self.__getUtilizationPM(idPMo,cvm,cbl,cbl_idf)
        uPMd = self.__getUtilizationPM(idPMd,cvm,cbl,cbl_idf)
        #Se calculan los máximos de PM
        uPMto = self.__getMaxUtilizationPM(idPMo)
        uPMtd = self.__getMaxUtilizationPM(idPMd)
    
        #La nueva utilización_Origen = actual - origen + destino
        newUPMo = uPMo-uVMo+uVMd
        
        if uPMto[0] < newUPMo[0]: return False
        if uPMto[2] < self.blockToTeras(newUPMo[2]): return False        
        if uPMto[1] < self.MbToGB(newUPMo[1]): return False        
    
        newUPMd = uPMd-uVMd+uVMo
    
        if uPMtd[0] < newUPMd[0]: return False
        if uPMtd[2] < self.blockToTeras(newUPMd[2]): return False        
        if uPMtd[1] < self.MbToGB(newUPMd[1]): return False    
        
        return True
    
    """ Para M2 """
    def __is_REMOVE_MV_suitable(self,vmIdo,vmIdd,setVM,cvm,cbl,cbl_idf):
        #La carga que soporta la máquina original es:
        uVMo = self.__demand(vmIdo,cbl,cbl_idf)       
        #Se calcula las utilizaciones actuales de las diferentes PM
        idPMd = cvm[vmIdd]
        uPMd = self.__getUtilizationPM(idPMd,cvm,cbl,cbl_idf)
        #Se calculan los máximos de PM
        uPMtd = self.__getMaxUtilizationPM(idPMd)
        #La nueva utilización_Destino = actual + origen
        newUPMd = uPMd+uVMo
        if uPMtd[0] < newUPMd[0]: return False
        if uPMtd[2] < self.blockToTeras(newUPMd[2]): return False        
        if uPMtd[1] < self.MbToGB(newUPMd[1]): return False    
        return True
    """ PARA M1 """   
    def __cbl_availabilityNewVM(self,pmi,cbl,cbli,cvm,cbl_idf):
        fcbl = cbl
        fcbl[cbli]=len(cvm)
        fcvm = np.append(cvm,[pmi])
        uPMo = self.__getUtilizationPM(pmi,fcvm,fcbl,cbl_idf)
        uPMto = self.__getMaxUtilizationPM(pmi)
        if uPMto[0] < uPMo[0]: return False
        if uPMto[2] < self.blockToTeras(uPMo[2]): return False        
        if uPMto[1] < self.MbToGB(uPMo[1]): return False    
        return True
    """ Para M4 """
    """ Eq. a M2 """
    def __is_CHANGE_MV_suitable(self,vmi,pmi,setVM,cvm,cbl,cbl_idf):
        #La carga que soporta la máquina original es:
        uVMo = self.__demand(vmi,cbl,cbl_idf)       
        #Se calcula las utilizaciones actuales de las diferentes PM
        uPMd = self.__getUtilizationPM(pmi,cvm,cbl,cbl_idf)
        #Se calculan los máximos de PM
        uPMtd = self.__getMaxUtilizationPM(pmi)
        #La nueva utilización_Destino = actual + origen
        newUPMd = uPMd+uVMo
        if uPMtd[0] < newUPMd[0]: return False
        if uPMtd[2] < self.blockToTeras(newUPMd[2]): return False        
        if uPMtd[1] < self.MbToGB(newUPMd[1]): return False    
        return True
    
    """ PARA M5 """
    def __getFRBlock(self,cbli,cbl):
        cbliRep = cbli / self.ce["replicationFactor"]
        vmCBLi = [] #Resto de VM dentro de ese mismo bloque
        for value in range(0,self.ce["replicationFactor"]):
            vmCBLi.append(cbl[cbliRep*self.ce["replicationFactor"]+value])    
        return vmCBLi
    
    def __is_SWAP_CBL_suitable(self,cbli,cblj,setVM,cvm,cbl,cbl_idf):
        cblF = cbl
        vmi = cbl[cbli]  #Se asgina al bloque la nueva VM
        vmj = cbl[cblj]
        cblF[cbli] = vmj
        cblF[cblj] = vmi
        
        #Se calcula la nueva demanda de cada VM
        uVMi = self.__demand(vmi,cblF,cbl_idf)       
        uVMj = self.__demand(vmj,cblF,cbl_idf)       
    
        #Se calcula la nueva demanda de cada PM
        pmi = cvm[vmi]
        pmj = cvm[vmj]
        uPMdi = self.__getUtilizationPM(pmi,cvm,cblF,cbl_idf)
        uPMdj = self.__getUtilizationPM(pmj,cvm,cblF,cbl_idf)
        
        #Se calculan los máximos de PM
        uPMti = self.__getMaxUtilizationPM(pmi)
        uPMtj = self.__getMaxUtilizationPM(pmj)
        
        newUPMi = uPMdi+uVMi
    
        if uPMti[0] < newUPMi[0]: return False
        if uPMti[2] < self.blockToTeras(newUPMi[2]): return False        
        if uPMti[1] < self.MbToGB(newUPMi[1]): return False    
        
        newUPMj = uPMdj+uVMj
        
        if uPMtj[0] < newUPMj[0]: return False
        if uPMtj[2] < self.blockToTeras(newUPMj[2]): return False        
        if uPMtj[1] < self.MbToGB(newUPMj[1]): return False    
    
        return True
        
        
    """ PARA M6 """
    def __is_CHANGE_CBL_suitable(self,cbli,vmi,setVM,cvm,cbl,cbl_idf):
        #La VM tendrá más carga
        #Esa carga la soporta su PMi
        cblF = cbl
        cblF[cbli] = vmi
        uVMo = self.__demand(vmi,cblF,cbl_idf)   
        pmi = cvm[vmi]
        uPMd = self.__getUtilizationPM(pmi,cvm,cblF,cbl_idf)
        #Se calculan los máximos de PM
        uPMtd = self.__getMaxUtilizationPM(pmi)
    
        #La nueva utilización_Destino = actual + origen
        newUPMd = uPMd+uVMo
    
        if uPMtd[0] < newUPMd[0]: return False
        if uPMtd[2] < self.blockToTeras(newUPMd[2]): return False        
        if uPMtd[1] < self.MbToGB(newUPMd[1]): return False    
        
        return True



    """ Potential cost of wasted resources """
    def __getWasted_PM(self,pmi,cvm,cbl,cbl_idf):
        epsilon = 0.15
        uPMd = self.__getUtilizationPM(pmi,cvm,cbl,cbl_idf)
        desv = np.std(uPMd)
        cores = self.ce["PM"][pmi]["Core"]
        uPMd[0] = uPMd[0]/cores
        if np.sum(uPMd)==0:
            return 0
        else:
            return (desv+epsilon)/np.sum(uPMd) 
    
    def __getPower_PM(self,pmi,cvm,cbl,cbl_idf):
        uPMd = self.__getUtilizationPM(pmi,cvm,cbl,cbl_idf)
    #    print "CPU : U: %s" %uPMd[0]
        p_cpu = (self.ce["PM"][pmi]["CPU_full"]-self.ce["PM"][pmi]["CPU_idle"])*uPMd[0]+self.ce["PM"][pmi]["CPU_idle"]

        #Power memory
        s_mod = 0 #EN MB
        vmPMi = np.where(cvm==pmi)
        for vmi in vmPMi[0]:
            blokVMi = np.where(cbl==vmi)
            for blocki in blokVMi[0]:
                #Se los ficheros que usa
                s_mod +=self.ce["fileU"][cbl_idf[blocki]][1] #Valor de la memoria que consume ese fichero
            
        s_mod = self.MbToGB(s_mod)
        n_mod = 1 
        ro = 0.9
        u_MM = self.MbToGB(uPMd[1])/ self.ce["PM"][pmi]["MM"] #capacidad
        p_mm = n_mod*s_mod*ro*u_MM  
    

        #Power HD
    #    print "HD: Teras Usados_HD: %s" %blockToTeras(uPMd[2])
        U_hd = self.blockToTeras(uPMd[2]) / self.ce["PM"][pmi]["HD"]
        p_hd = (1+0.4*U_hd)*self.ce["PM"][pmi]["HD_idle"]
        return p_cpu+p_mm+p_hd

        
    def __getFailure_System(self,cvm,cbl,cbl_idf,setVM):
        failureVM = np.linspace(-1,-1,len(cvm))
        failurePM = np.linspace(-1,-1,self.ce["numberOfPM"])
        for vmi in range(len(cvm)):
            uVMi = self.__demand(vmi,cbl,cbl_idf)
            uVMi_1 = uVMi[0]/setVM[vmi]["Core"]
            failureVM[vmi] = np.linspace(setVM[vmi]["failure"][0],setVM[vmi]["failure"][1],100)[int(uVMi_1*100)]

        for pmi in range(self.ce["numberOfPM"]):
             uPMi = self.__getUtilizationPM(pmi,cvm,cbl,cbl_idf)
             uPMi_1 =  uPMi[0] / self.ce["PM"][pmi]["Core"]
             failurePM[pmi] = np.linspace(self.ce["PM"][pmi]["failure"][0],self.ce["PM"][pmi]["failure"][1],100)[int(uPMi_1*100)]


        prevMul = 1
        failureTotal = 0.0
        setPM = set()
        idx =0
        for vmi in cbl:
           # print idx
            prevMul*= failureVM[vmi]
            setPM.add(cvm[vmi])
            if idx%self.ce["replicationFactor"] == self.ce["replicationFactor"]-1:
                    failureTotal += prevMul
                    failureTotalPM = 1
                    for pmi in setPM:
                        failureTotalPM *=failurePM[pmi]
                    failureTotal += failureTotalPM
                    setPM = set()
                    prevMul = 1

            idx += 1
        return failureTotal
        
        
    """ Get Fitness values of Citizen """
    def get_fitness_citizen(self,cvm,cbl,cbl_idf,setVM):
        fit = np.zeros((self.ce["numberOfPM"],2))
        #print "Entrada FIT Citizen"
        for pmi in range(self.ce["numberOfPM"]):
#            cores = self.ce["PM"][pmi]["Core"]
#            print pmi,cores
            #print "W"
            w= self.__getWasted_PM(pmi,cvm,cbl,cbl_idf)
            #print "E_W"
            
            if w==0.0: #Alpha factor formula 9 y 10
                p = 0.0
            else:
                p = self.__getPower_PM(pmi,cvm,cbl,cbl_idf)
            #print "E_PM      "
            fit[pmi] = [w, p]

        f = self.__getFailure_System(cvm,cbl,cbl_idf,setVM)

        #TODO sumatorio fit
        return [np.sum(fit[:,0]),np.sum(fit[:,1]),f]
        
        
    def getCitizen(self,i):
        return self.pop[i]
        
        
    """ Creating initial Citizens-Chromosomes """
    # size - of population
    def population(self,size):
        self.logger.info('**** Population: %s' %size)
        self.population = size
        sizeFiles = self.ce["sizeFiles"]
        totalBlocksWR = np.sum(sizeFiles)*self.ce["replicationFactor"]
        setVM = self.__getMinSetVM()

        self.logger.info('\t Initial Estimation  of virtal machines: %i' %len(setVM))
#        print "Número de máquinas virtuales min: %i" %len(setVM)
        
        self.fit = np.zeros(size)
        #create citizens        
        for idCiti in range(size):
#            print "Creando citizen: %i" %idCiti
            self.logger.info('\t Creating Citizen: %i' %idCiti)
            
            #Asignación de bloques a MV
#            print "\tAsignación de BLQ a MV"
            self.logger.info('\t\t Adding BLQ to VM')
            cbl = np.linspace(-1,-1,totalBlocksWR).astype(int) #un array init. de -1
            cvm_u = np.zeros((len(setVM),3))  # un array init. de -1
            cbl_idf = np.linspace(-1,-1,totalBlocksWR).astype(int)
            idc = 0
            for idFile,blockF in enumerate(sizeFiles): #para cada file
                previousVMI=[]
                for block in range(blockF*self.ce["replicationFactor"]): #para el numero total de bloques
                     if block%self.ce["replicationFactor"]==0: previousVMI=[]
                     vmi = random.randint(0,len(setVM)-1)
                     tries = self.__defaultTries
                     while vmi in previousVMI  or not self.__cbl_availability(vmi,setVM,cbl,cbl_idf,idFile,cvm_u):
                         vmi = random.randint(0,len(setVM)-1)
                         tries-=1
                         if tries==0:
                             self.logger.info('\t\t ADDING a new VM')
                             setVM = list(setVM)+list(np.random.choice(self.casesOfVM,1))
                             cvm_u = np.vstack((cvm_u,[0,0,0]))
                             tries = self.__defaultTries
             
                     cbl[idc]= vmi 
                     cbl_idf[idc] = idFile
                     idc+=1
                     previousVMI.append(vmi)
                     cvm_u[vmi] = np.add(self.ce["fileU"][idFile]/self.ce["replicationFactor"],cvm_u[vmi])

            self.logger.info('\t\t Final number of virtal machines: %i' %len(setVM))
#            if len(np.where(cbl==len(setVM)))>0: print "[ERROR] Desconocido cbl[j]= VMi>Len(setVM)"

            #Asignación de MV a PM
            self.logger.info('\t\t Asignación de MV a MP')
            totalVM = len(setVM)
            totalPM = self.ce["numberOfPM"]
            cvm = np.linspace(-1,-1,totalVM).astype(int)
            for idx in range(totalVM):
                pmi = random.randint(0,totalPM-1)
                tries = self.__defaultTries
                while not self.__space_availabilityPM(pmi,cvm,idx,setVM,cbl,cbl_idf):   
                    pmi = random.randint(0,totalPM-1)
                    print pmi
                    tries -=1
                    if tries==0: 
                             self.logger.error('\t\t No más MP disponibles ') #ESTO NO DEBERIA DE PASAR
                             raise NameError('MP-no disponibles')
                cvm[idx]= pmi              
            
            #Fitnes value
            #print "Saliendo assginacin MV a PM"
            fit_value = self.get_fitness_citizen(cvm,cbl,cbl_idf,setVM)
 
            #Un ciudadano tiene un CVM, CBL, SETVM            
            self.pop[idCiti]={"cvm":cvm,"cbl":cbl,"setVM":setVM,"cbl_idf":cbl_idf,
                              "fit_value":fit_value}
            #print "Saliendo fitness"
         
        #endFor citizens
        #Fremte pareto
        self.fitnessGeneration = self.get_Fitness_value()
                   
        #TODO PENDIENTE
        
    #==============================================================================
    # #FRENTE PARETO
    #==============================================================================
    def get_Fitness_value(self):        
        fit_values = []
        for idx in range(self.population):
            fit_values.append(self.pop[idx]["fit_value"])
        fit_values = np.array(fit_values)
        
        paretoPoints,dominatedPoints = self.cull(fit_values.tolist(),self.dominates)
        frentePareto = np.array(paretoPoints)    

        print "\t Frente Pareto size: %i" %len(frentePareto)
    

        Wmax = np.max(frentePareto[:,0])
        Wmin = np.min(frentePareto[:,0])
        Pmax = np.max(frentePareto[:,1])
        Pmin = np.min(frentePareto[:,1])
        Fmax = np.max(frentePareto[:,2])
        Fmin = np.min(frentePareto[:,2])

        Wmean = np.mean(frentePareto[:,0])     
        Pmean = np.mean(frentePareto[:,1])
        Fmean = np.mean(frentePareto[:,2])

        if Fmax == Fmin :
            Fmax = 1
            Fmin = 0


        if Wmax == Wmin :
            Wmax = 1
            Wmin = 0

        if Pmax == Pmin :
            Pmax = 1
            Pmin = 0

        self.fit = []

        for idx,cit in enumerate(self.pop):
                self.fit.append((self.pop[idx]["fit_value"][0] / (3*  (Wmax-Wmin))+
                                     self.pop[idx]["fit_value"][1] / (3 * (Pmax-Pmin))+
                                     self.pop[idx]["fit_value"][2] / (3 * (Fmax-Fmin))))

        #TODO NO TENGO CLARO ESTO....
        return  [Wmax,Wmin,Pmax,Pmin,Fmax,Fmin,Wmean,Pmean,Fmean,np.min(self.fit)]

        

    def dominates(self,row, rowCandidate):
        return all(r >= rc for r, rc in zip(row, rowCandidate))
    
    def cull(self,pts, dominates):
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
        
    
    """ MUTATIONS on any chromosome  """
    """ M1 -create a new Virtual Machine  """
    def __M1_createVM(self,citizen):
        self.logger.info("\t\t Type of Matutation: M1 ")
        setVM = citizen["setVM"]
        cvm = citizen["cvm"]
        cbl = citizen["cbl"]
        cbl_idf = citizen["cbl_idf"]
        
        vmi = random.randint(0,len(self.casesOfVM)-1)
        pmi = random.randint(0,self.ce["numberOfPM"]-1)
        cbli = random.randint(0,len(cbl)-1)
        tries = self.__defaultTries
        previousPMI = []
        allOK=True
        while (pmi in previousPMI or not self.__cbl_availabilityNewVM(pmi,cbl,cbli,cvm,cbl_idf)):
            previousPMI.append(pmi)
            pmi = random.randint(0,self.ce["numberOfPM"]-1)
            tries-=1
            if tries==0 or len(np.unique(previousPMI))==self.ce["numberOfPM"]: 
                allOK = False
#                print "ERROR - M1 - No hay posibilidad de añadir una nueva VM a un PM"
                return None,"M1-Create VM: Fallido" 
        if allOK: 
#            print "OK"
            self.logger.debug("\t\t\t New VM on PMi: %i" %pmi)
            setVM.append(self.casesOfVM[vmi])
            cbl[cbli]=len(cvm)
            cvm = np.append(cvm,[pmi])
            citizen["setVM"] = setVM
            citizen["cvm"] = cvm
            citizen["cbl"] = cbl
            return citizen,"M1-Create_VM: OK"
        else:
            return None,"M1-Create VM: Fallido" 
    
        
    """ M2 -remove a Virtual Machine """        
    def __M2_removeVM(self,cit):
        self.logger.info("\t\t Type of Matutation: M2 ")
        setVM = cit["setVM"]
        cvm = cit["cvm"]
        cbl = cit["cbl"]
        cbl_idf = cit["cbl_idf"]

        if len(cvm)==0:
            self.logger.warning("\t\t\t Remove VM - No hay más máquinas")
            return None,"M2-REMOVE_VM: No Dispone de más maquinas" 
           
        vmi = random.randint(0,len(cvm)-1)
        vmd = random.randint(0,len(cvm)-1)
        
        #Se eligen origen y destino
        previousSelection = []
        while ( setVM[vmd]["id"] in previousSelection
               or vmi == vmd
               or setVM[vmd]["id"] ==setVM[vmi]["id"] 
               or not self.__is_REMOVE_MV_suitable(vmi,vmd,setVM,cvm,cbl,cbl_idf)):
            previousSelection.append(setVM[vmd]["id"])
            vmd = random.randint(0,len(cvm)-1)
            if len(previousSelection)==len(self.casesOfVM):
#                print "[Warning] No hay ninguna VM-PM que soporte: M2-remove"
                vmd = -1
                return None,"M2-REMOVE_VM: Fallido"
     
        if vmd >=0: #Todo ha ido bien
#            print "REMOVE OK MV:%i  =-> MV:%i " %(vmi,vmd)
            self.logger.debug("\t\t\t Remove VMo: %i" %vmi)
            self.logger.debug("\t\t\t  -- movVMd: %i" %vmd)
            np.place(cbl,cbl==vmi,vmd)#Al destino se le asignan los bloques de Ori
            cvm = np.delete(cvm,vmi)

            idxBloq = np.where(cvm>vmi)
            for i in idxBloq:
                cbl[i] -= 1

            del setVM[vmi]
            cit["cvm"] = cvm
            cit["cbl"] = cbl
            cit["setVM"] = setVM
            return cit,"M2-REMOVE_VM: OK"
        else:
            return None,"M2-REMOVE_VM: Fallido"


        
    """ M3 -swap a Virtual Machine """   
    def __M3_swapVM(self,cit):
        self.logger.info("\t\t Type of Matutation: M3 ")
        setVM = cit["setVM"]
        cvm = cit["cvm"]
        cbl = cit["cbl"]
        cbl_idf = cit["cbl_idf"]
        totalVM = len(setVM)
        #Se eligen origen y destino
        orig = random.randint(0,totalVM-1) #La posición 22, tiene una máquina virtual del tipo: cvm[orig]["id"]
        dest = random.randint(0,totalVM-1)
        previousSelection = []
        while (setVM[dest]["id"] in previousSelection
               or orig == dest
               or  setVM[dest]["id"] ==setVM[orig]["id"] 
               or not self.__is_SWAP_MV_suitable(orig,dest,setVM,cvm,cbl,cbl_idf)):
            previousSelection.append(setVM[dest]["id"])
            dest = random.randint(0,totalVM-1)
            if len(previousSelection)==len(self.casesOfVM):
#                print "[Warning] No hay ninguna VM-PM que soporte: M3-swap"
                dest = -1
                return None,"M3-SWAP_VM: Fallido"
        if dest >=0: #Todo ha ido bien
#            print "Swap OK: MV:%i  <-=-> MV:%i " %(orig,dest)
            self.logger.debug("\t\t\t SWAP VMo: %i <--> VMd: %i" %(orig,dest))
            #En cada se intercambia el ID de la VM
            np.place(cbl,cbl==orig,-1)
            np.place(cbl,cbl==dest,-2)
            np.place(cbl,cbl==-1,dest)
            np.place(cbl,cbl==-2,orig)

            cit["cbl"] = cbl
            return cit,"M3-SWAP_VM: OK"
        else:
            return None,"M3-SWAP_VM: Fallido"
        
    """ M4 - Move Virtual Machine """   
    def __M4_MoveVM(self,cit):
        self.logger.info("\t\t Type of Matutation: M4 ")
        setVM = cit["setVM"]
        cvm = cit["cvm"]
        cbl = cit["cbl"]
        cbl_idf = cit["cbl_idf"]

        vmi = random.randint(0,len(setVM)-1)
        pmi = random.randint(0,self.ce["numberOfPM"]-1)
        
#        print "VMI: %i --->  %d " %(vmi,pmi)
        previousSelection = []
        while (pmi in previousSelection #No se encontraba alla
               or pmi == cvm[vmi] #Ya estaba asignada esa VM a PM
               or not self.__is_CHANGE_MV_suitable(vmi,pmi,setVM,cvm,cbl,cbl_idf)):
            previousSelection.append(pmi)
            pmi = random.randint(0,self.ce["numberOfPM"]-1)
            if len(np.unique(previousSelection))==self.ce["numberOfPM"]: #D1
#                print "[Warning] M4. No hay ninguna PM que soporte: Vmi-Change"
                pmi = -1
                return None,"M4-Move_VM: Fallido"
        
        if pmi >=0:
#            print pmi
            self.logger.debug("\t\t\t MOVE VMo %i  :--> Pmi: %i" %(vmi,pmi))
            cvm[vmi] = pmi #Se asgina la CVM a PMI
            cit["cvm"] = cvm
            return cit,"M3-Move_VM: Ok"
        else:
            return None,"M4-Move_VM: Fallido"

        

     
    """ M5 - swap a CBL """   
    def __M5_swapCBL(self,cit):   
        self.logger.info("\t\t Type of Matutation: M5 ")
        setVM = cit["setVM"]
        cvm = cit["cvm"]
        cbl = cit["cbl"]
        cbl_idf = cit["cbl_idf"]
        
        cbli = random.randint(0,len(cbl)-1)
        vmCBLi = self.__getFRBlock(cbli,cbl) 
          
        cblj = random.randint(0,len(cbl)-1)
        vmCBLj = self.__getFRBlock(cblj,cbl)
        
        z = np.hstack((vmCBLi,vmCBLj))
        tries = self.__defaultTries
        cjj=-1
        while cjj<0 and tries>0:
#            print "CBLi: %i ----> CBLj: %i" %(cbli,cblj)
            while (not self.ce["replicationFactor"]*2==len(np.unique(z))):
                cblj = random.randint(0,len(cbl)-1)
                vmCBLj = self.__getFRBlock(cblj,cbl)
                z = np.hstack((vmCBLi,vmCBLj))
                if tries==0:
                    break
                tries-=1
            #Combinación de bloques factibles
            #Puedo cambiar ALGUNO de los tres bloques de vmBCLj
            if tries >0:
                for cblj in vmCBLj:
                    if self.__is_SWAP_CBL_suitable(cbli,cblj,setVM,cvm,cbl,cbl_idf):
#                        print "OK"
                        cjj = cblj
                        break
                    
                if cjj<0:
                    z=[]
                    
        if cjj>0:
            self.logger.debug("\t\t\t SWAP CBLi: %i <--> CBLj: %i" %(cbli,cjj))
            vmi = cbl[cbli]  #Se asgina al bloque la nueva VM
            cbl[cbli] = cbl[cjj]
            cbl[cjj] = vmi
            cit["cbl"] = cbl
            return cit,"M5-SWAP_CBL: Ok"
        else:
            return None,"M5-SWAP_CBL: Fallido"
    
        
    """ M6 - Move value CBL """   
    def __M6_MoveCBL(self,cit):
        self.logger.info("\t\t Type of Matutation: M6 ")
        setVM = cit["setVM"]
        cvm = cit["cvm"]
        cbl = cit["cbl"]
        cbl_idf = cit["cbl_idf"]
        cbli = random.randint(0,len(cbl)-1)
        #Control del factor de replicacion
        cbliRep = cbli / self.ce["replicationFactor"]
        VmReplicactionCBLValues = [] #Resto de VM dentro de ese mismo bloque
        for value in range(0,self.ce["replicationFactor"]):
            VmReplicactionCBLValues.append(cbl[cbliRep*self.ce["replicationFactor"]+value])
        
        #La nueva VM no ha de estar dentro de los valores de replicacion anteriores, ya se incluye la propia maquina
        vmi = random.randint(0,len(setVM)-1)
#        print "CBLi: %i ---> VM: %d " %(cbli,vmi)
        previousSelection = VmReplicactionCBLValues
        while (vmi in previousSelection #No se encontraba alla
               or not self.__is_CHANGE_CBL_suitable(cbli,vmi,setVM,cvm,cbl,cbl_idf)):
            previousSelection.append(vmi)
            vmi = random.randint(0,len(setVM)-1)
            if len(np.unique(previousSelection))==len(setVM): #D1
#                print "[Warning] M6. No hay ninguna VM que soporte: CBLi-Change"
                vmi = -1
                return None,"M6-Move_CBL: Fallido"
        
        if vmi >=0:
            self.logger.debug("\t\t\t MOVE CBLi %i  :--> VMi: %i" %(cbli,vmi))
#            print vmi
            cbl[cbli] = vmi #Se asgina al bloque la nueva VM
            cit["cbl"] = cbl
            return cit,"M6-Move_CBL: Ok"
        else:
            return None,"M6-Move_CBL: Fallido"

      
    def mutate(self):
#        mutations = [self.__M2_removeVM]
        self.logger.info("*** Matutation ")
        mutations = [self.__M1_createVM,self.__M2_removeVM,self.__M3_swapVM,self.__M4_MoveVM,
                     self.__M5_swapCBL,self.__M6_MoveCBL]
        citi = random.randint(0,self.population-1)
#        print "Ciudadano a mutar: %i" %citi
        self.logger.info("\t Citizen: %i "%citi)
        cit = self.pop[citi]
        muti = random.randint(0,len(mutations)-1)
        citM,state =  mutations[muti](cit)
        if citM !=None:
            self.logger.info("\t\t Mutation status: OK")
            self.pop[citi] = citM 
        else:
            self.logger.info("\t\t Mutation status: FAILED")
        return state
    
    """ CROSSOVERS """
    def C1_OneCuttingPoint(self,citOri,citDest):
        citM = self.pop[citOri]
        citF = self.pop[citDest]
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


    def C2_TwoCuttingPoint(self,citOri,citDest):
        citM = self.pop[citOri]
        citF = self.pop[citDest]
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
    
    def C3_NonUniform(self,citOri,citDest):
        citM = self.pop[citOri]
        citF = self.pop[citDest]
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

   
      
    """ Evolve operation """
    def evolve(self,crossoverfunction,crossover_treshold,tries):
        self.logger.info("*** Evolution")
        #Preparing probabilities of each citizen

        candidates = np.sort(self.fit)
        p_candidates = []
        p_cand = []
        for value in candidates: p_candidates.append(1/(value/np.sum(candidates)))
        for value in p_candidates: p_cand.append(value/np.sum(p_candidates))
        
        #Creación de una nueva población
        citizenid = 0
        allOk = tries
        previousNoFactibleFathers = []
        future_population = {}
        while citizenid<self.population and allOk>=0:
            #==============================================================================
            # #Seleccion: Roulette-wheel selection
            #==============================================================================
            #Los mejores están mas cerca del cero
            candys = np.random.choice(candidates,2,replace=False,p=p_cand)
            citOri = np.where(self.fit==candys[0])[0][0]
            citDest = np.where(self.fit==candys[1])[0][0]
            if not [citOri,citDest] in previousNoFactibleFathers:
                self.logger.debug("\t Crossing parents: %i - %i" %(citOri,citDest))
#                print "Crossing parents: %i - %i" %(citOri,citDest)
                #Applying one crossover

                citM = self.pop[citOri]
                citF = self.pop[citDest]        
                if random.random()<crossover_treshold:
                    #==============================================================================
                    # CrossOverS
                    #==============================================================================
                    h1_cvm,h1_cbl,h1_cbl_idf,h1_setVM,h2_cvm,h2_cbl,h2_cbl_idf,h2_setVM = crossoverfunction(citOri,citDest)
                        
                    possible = True
                    for pmi in range(self.ce["numberOfPM"]):
                        upmih1 = self.__getUtilizationPM(pmi,h1_cvm,h1_cbl,h1_cbl_idf)
                        upmih2 = self.__getUtilizationPM(pmi,h2_cvm,h2_cbl,h2_cbl_idf)
                        uMax = self.__getMaxUtilizationPM(pmi)
                        if uMax[0] < upmih1[0]: possible= False
                        if uMax[2] < self.blockToTeras(upmih1[2]): possible = False        
                        if uMax[1] < self.MbToGB(upmih1[1]): possible = False    
                        if uMax[0] < upmih2[0]: possible= False
                        if uMax[2] < self.blockToTeras(upmih2[2]): possible = False        
                        if uMax[1] < self.MbToGB(upmih2[1]): possible = False   
                        if not possible: break
                    
                    if possible:
                        self.logger.debug("\t Creating two new citizens")
                        fitness_values1 = self.get_fitness_citizen(h1_cvm,h1_cbl,h1_cbl_idf,h1_setVM)
                        future_population[citizenid] = {"cvm":h1_cvm,"cbl":h1_cbl,"setVM":h1_setVM,
                                                    "cbl_idf":h1_cbl_idf, "fit_value":fitness_values1}
                    
                        citizenid+=1
                        fitness_values2 = self.get_fitness_citizen(h2_cvm,h2_cbl,h2_cbl_idf,h2_setVM)
                        future_population[citizenid] = {"cvm":h2_cvm,"cbl":h2_cbl,"setVM":h2_setVM,
                                                    "cbl_idf":h2_cbl_idf,"fit_value":fitness_values2}
                        citizenid+=1  
                    else:
                        previousNoFactibleFathers.append([citOri,citDest])                         
                else: #No crossover
                    if not np.any(future_population.values() == citM) and not np.any(future_population.values() == citF):
                        self.logger.debug("\t Populating with parents")
                        future_population[citizenid] = citM
                        citizenid+=1
                        future_population[citizenid] = citF
                        citizenid+=1 
             #Padres anteriormente contrastados           
            else: 
                allOk-=1 #Evitar bucle infinito

        #Endwhile        
        #Se sustituye la población actual por la nueva 
        if allOk>=0:
            self.pop =  future_population
            #Calculamos fitness de la poblacion
            self.fitnessGeneration = self.get_Fitness_value()
            return True
        else:
            return False
       #print future_population
#        print len(future_population)     
#        print allOk
 
            
         
    """ Print Population """
    def show(self):
        for idx in self.pop.keys():
            print "Citizen: %i" %idx
            print "\tsetVM:\t%s" %self.pop[idx]["setVM"]
            print "\tCBL:\t%s" %self.pop[idx]["cbl"]
            print "\tCBL_IF:\t%s" %self.pop[idx]["cbl_idf"]
            print "\tCVM:\t%s" %self.pop[idx]["cvm"]