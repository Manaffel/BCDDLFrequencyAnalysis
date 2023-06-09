import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import psd
from matplotlib.pyplot import csd
from scipy.optimize import minimize
from scipy.optimize import curve_fit

class BCDDLFrequencyAnalysis():
    def __init__(self,T,X,Y,ZeroTime=None):
        """ Tool to analyse BCDDL data. Uses the measured output response Y for an input signal X 
            measured at times T.
        Parameters
        ----------
        T : M-shaped numpy array float [s]
            Quasi equidistant time points at which signals X and Y are measured
        X : M-shaped numpy array float [A.U.]
            Input signal measured at times T
        Y : M-shaped numpy array float [A.U.]
            Output signal measured at times T
        """
        self.T = T
        self.X = X
        self.Y = Y
        
        if ZeroTime == True:
            self.T = self.T-self.T[0]
            
        self.Data = np.array([self.T,self.X,self.Y])
        
    def plotData(self,tmin=None,tmax=None): 
        """ Plots the in- and output signals as a function of time for time in [tmin,tmax].
        Parameters
        ----------
        tmin :  float [s]
                Lower time limit
        tmax :  float [s]
                Upper time limit
        """
        #Limit signals to [tmin,tmax]
        Data = self.Data
        if tmin != None:
            Data = Data[:,Data[0,:]>=tmin]            
        if tmax != None:
            Data = Data[:,Data[0,:]<=tmax]
        
        #Plot signals
        plt.figure()
        plt.plot(Data[0,:],Data[1,:],label='X')
        plt.plot(Data[0,:],Data[2,:],label='Y')
        
        plt.legend(loc='best',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Time [s]',fontsize=14)
        plt.ylabel('Signal [A.U.]',fontsize=14)
        plt.tight_layout()
        plt.show()
        
    def plotCoherence(self,nmin,nmax,tmin=None,tmax=None):
        """ Plots the coherence within the signals as a function of frequency for differnt 
            numbers of subsamples. The subsamples are used to average over the coherence.
        ----------
        nmin :  int [-]
                Lower subsample range (2**nmin)            
        nmin :  int [-]
                Upper subsample range (2**nmax) 
        tmin :  float [s]
                Lower time limit
        tmax :  float [s]
                Upper time limit
        """
        #Limit signals to [tmin,tmax]
        Data = self.Data
        if tmin != None:
            Data = Data[:,Data[0,:]>=tmin]            
        if tmax != None:
            Data = Data[:,Data[0,:]<=tmax]
        
        #Average Frequency
        Fs = 1/np.mean(Data[0,1:]- Data[0,:-1])        
        #Number of Subsamples
        N = 2**np.linspace(nmin,nmax,nmax-nmin+1,dtype=int)
        coh = []
        fr = []
        
        #Calculate coherence for different sample numbers
        fig = plt.figure()
        for n in N:
            NFFT=len(Data[0,:])//n
            Pxy,f = csd(Data[1,:],Data[2,:],NFFT=NFFT,Fs=Fs)
            Pxx,f = psd(Data[1,:],NFFT=NFFT,Fs=Fs)
            Pyy,f = psd(Data[2,:],NFFT=NFFT,Fs=Fs)
            coh.append(np.abs(Pxy)**2/(Pxx*Pyy))
            fr.append(f)
        plt.close(fig)
        
        #Plot coherence
        plt.figure()
        for i in range(len(N)):
            plt.plot(fr[i],coh[i],label=str(N[i]))
            
        plt.xscale('log')
        plt.legend(loc='best',fontsize=12,title='Subsamples:')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel('Coherence [-]',fontsize=14)
        plt.xlabel('Frequency [Hz]',fontsize=14)
        plt.tight_layout()
        plt.show()
        
    def plotFRF(self,N,tmin=None,tmax=None,fmin=None,fmax=None):
        """ Plots the bode plots of the FRF and the coherence as function of frequency for 
            frequencies in [fmin,fmax].
        Parameters
        ----------
        N    :  int [-]
                Number of subsamples
        tmin :  float [s]
                Lower time limit
        tmax :  float [s]
                Upper time limit
        fmin :  float [Hz]
                Lower frequency limit
        fmax :  float [Hz]
                Upper frequency limit
        """
        #Limit signals to [tmin,tmax]
        Data = self.Data
        if tmin != None:
            Data = Data[:,Data[0,:]>=tmin]            
        if tmax != None:
            Data = Data[:,Data[0,:]<=tmax]
        
        #Average Frequency
        Fs = 1/np.mean(Data[0,1:]- Data[0,:-1])  
        #Subsample length
        NFFT=len(Data[0,:])//N
        
        #Calculate coherence
        fig = plt.figure()
        Pxy,f = csd(Data[1,:],Data[2,:],NFFT=NFFT,Fs=Fs)
        Pxx,f = psd(Data[1,:],NFFT=NFFT,Fs=Fs)
        Pyy,f = psd(Data[2,:],NFFT=NFFT,Fs=Fs)
        plt.close(fig)        
        coh = np.abs(Pxy)**2/(Pxx*Pyy)
        
        #Calculate FRF
        H = Pxy/Pxx
        #Limit data to [fmin,fmax]
        if fmin != None:
            H = H[f>=fmin]
            coh = coh[f>=fmin]
            f = f[f>=fmin]
        if fmax != None:
            H = H[f<=fmax]
            coh = coh[f<=fmax]
            f = f[f<=fmax]
        
        #Calculate magnitude and phase
        mag = 20*np.log10(np.abs(H))
        ph = np.angle(H,deg=True)
        
        #Plot data
        fig,ax = plt.subplots(3,1,figsize=(8,6),sharex=True)
        ax[0].plot(f,mag,linewidth=3)
        ax[1].plot(f,ph,linewidth=3)
        ax[2].plot(f,coh,linewidth=3)
        
        ax[0].set_ylabel('Magnitude [dB]',fontsize=14)
        ax[1].set_ylabel('Phase [°]',fontsize=14)
        ax[2].set_ylabel('Coherence [-]',fontsize=14)
        ax[2].set_xlabel('Frequency [Hz]',fontsize=14)
        ax[2].set_xscale('log')
        fig.tight_layout()
        plt.show()
    
    def fitLowpass(self,N,f0,Q0,T0,tmin=None,tmax=None,fmin=None,fmax=None,ffitmax=None):
        """ Plots the bode plots of the FRF and the coherence as function of frequency for 
            frequencies in [fmin,fmax] and fits a 2nd order lowpass with delay up to frequencies 
            <= ffitmax.
        Parameters
        ----------
        N    :  int [-]
                Number of subsamples
        f0   :  float [Hz]
                Initial cut-off frequency guess for the fit 
        Q0   :  float [-]
                Initial quality factor guess for the fit 
        T0   :  float [s]
                Initial delay guess for the fit 
        tmin :  float [s]
                Lower time limit
        tmax :  float [s]
                Upper time limit
        fmin :  float [Hz]
                Lower frequency limit
        fmax :  float [Hz]
                Upper frequency limit
        """
        #Limit signals to [tmin,tmax]
        Data = self.Data
        if tmin != None:
            Data = Data[:,Data[0,:]>=tmin]            
        if tmax != None:
            Data = Data[:,Data[0,:]<=tmax]
        
        #Average frequency
        Fs = 1/np.mean(Data[0,1:]- Data[0,:-1])  
        #Subsample length
        NFFT=len(Data[0,:])//N
        
        #Calculate coherence
        fig = plt.figure()
        Pxy,f = csd(Data[1,:],Data[2,:],NFFT=NFFT,Fs=Fs)
        Pxx,f = psd(Data[1,:],NFFT=NFFT,Fs=Fs)
        Pyy,f = psd(Data[2,:],NFFT=NFFT,Fs=Fs)
        plt.close(fig)        
        coh = np.abs(Pxy)**2/(Pxx*Pyy)
        
        #Calculate FRF
        H = Pxy/Pxx
        
        #limit data to [fmin,fmax]
        if fmin != None:
            H = H[f>=fmin]
            coh = coh[f>=fmin]
            f = f[f>=fmin]
        if fmax != None:
            H = H[f<=fmax]
            coh = coh[f<=fmax]
            f = f[f<=fmax]
        
        #Calculate magnitude and phase
        mag = 20*np.log10(np.abs(H))
        ph = np.angle(H,deg=True)
        
        #Calculate lowpass fit
        if ffitmax:
            ffit = f[f<=ffitmax]
            Hfit = H[f<=ffitmax]
        else:
            ffit = f
            Hfit = H
        resx = estimateLowpass(ffit,Hfit,[f0,Q0,T0])
        print('Fit Parameters: ',resx)
        Hfit = Lowpass(resx,ffit)
        magfit = 20*np.log10(np.abs(Hfit))
        phfit = np.angle(Hfit,deg=True)
        
        #Plot data
        fig,ax = plt.subplots(3,1,figsize=(8,6),sharex=True)
        ax[0].plot(f,mag,linewidth=3,label='FRF')
        ax[0].plot(ffit,magfit,'--',color='black',linewidth=3,label='Lowpass Fit')
        ax[1].plot(f,ph,'--',linewidth=3)
        ax[1].plot(ffit,phfit,'--',color='black',linewidth=3)
        ax[2].plot(f,coh,linewidth=3)
        
        ax[0].legend(loc='best',fontsize=12)
        ax[0].set_ylabel('Magnitude [dB]',fontsize=14)
        ax[1].set_ylabel('Phase [°]',fontsize=14)
        ax[2].set_ylabel('Coherence [-]',fontsize=14)
        ax[2].set_xlabel('Frequency [Hz]',fontsize=14)
        ax[2].set_xscale('log')
        fig.tight_layout()
        plt.show()
    
    def plotTFFit(self,N,num,den,T0,tmin=None,tmax=None,fmin=None,fmax=None):
        """ Plots the bode plots of the FRF and the coherence as function of frequency for 
            frequencies in [fmin,fmax] and fits the delay to the TF model according to the applied Numerator and Denomnator parameters
        Parameters
        ----------
        N    :  int [-]
                Number of subsamples
        num  :  np shaped numpy array int [-]
                Numerator polynomial parmeters 
        den   : np shaped numpy array int [-]
                Denominator polynomial parameters
        T0   :  float [s]
                Initial delay guess for the delay fit 
        tmin :  float [s]
                Lower time limit
        tmax :  float [s]
                Upper time limit
        fmin :  float [Hz]
                Lower frequency limit
        fmax :  float [Hz]
                Upper frequency limit
        """
        #Limit signals to [tmin,tmax]
        Data = self.Data
        if tmin != None:
            Data = Data[:,Data[0,:]>=tmin]            
        if tmax != None:
            Data = Data[:,Data[0,:]<=tmax]
        
        #Average frequency
        Fs = 1/np.mean(Data[0,1:]- Data[0,:-1])  
        #Subsample length
        NFFT=len(Data[0,:])//N
        
        #Calculate coherence
        fig = plt.figure()
        Pxy,f = csd(Data[1,:],Data[2,:],NFFT=NFFT,Fs=Fs)
        Pxx,f = psd(Data[1,:],NFFT=NFFT,Fs=Fs)
        Pyy,f = psd(Data[2,:],NFFT=NFFT,Fs=Fs)
        plt.close(fig)        
        coh = np.abs(Pxy)**2/(Pxx*Pyy)
        
        #Calculate FRF
        H = Pxy/Pxx
        
        #limit data to [fmin,fmax]
        if fmin != None:
            H = H[f>=fmin]
            coh = coh[f>=fmin]
            f = f[f>=fmin]
        if fmax != None:
            H = H[f<=fmax]
            coh = coh[f<=fmax]
            f = f[f<=fmax]
        
        #Calculate magnitude and phase
        mag = 20*np.log10(np.abs(H))
        ph = np.angle(H,deg=True)
        
        #Calculate delay fit
        resx = Delayestimate(f,H,T0,num,den)
        print('Delay fit: ',resx)
        Hfit = TFDelay(resx,f,num,den)
        magfit = 20*np.log10(np.abs(Hfit))
        phfit = np.angle(Hfit,deg=True)
        
        #Plot data
        fig,ax = plt.subplots(3,1,figsize=(8,6),sharex=True)
        ax[0].plot(f,mag,linewidth=3,label='FRF')
        ax[0].plot(f,magfit,'--',color='black',linewidth=3,label='TF Fit')
        ax[1].plot(f,ph,'--',linewidth=3)
        ax[1].plot(f,phfit,'--',color='black',linewidth=3)
        ax[2].plot(f,coh,linewidth=3)
        
        ax[0].legend(loc='best',fontsize=12)
        ax[0].set_ylabel('Magnitude [dB]',fontsize=14)
        ax[1].set_ylabel('Phase [°]',fontsize=14)
        ax[2].set_ylabel('Coherence [-]',fontsize=14)
        ax[2].set_xlabel('Frequency [Hz]',fontsize=14)
        ax[2].set_xscale('log')
        fig.tight_layout()
        plt.show()

def Lowpass(x,f):
    """ 2nd order lowpass with delay TF.
    Parameters
    ----------
    x : 3-shaped array float [Hz,-,s]
        TF parameters [fc,Q,T]
    f : N-shaped array float [Hz]
        Frequencies at which TF is evaluated
    Returns
    ----------
    risp :  N-shaped array float [-]
            TF values evaluated at frequencies f 
    """
    s = 1j*f*2*np.pi
    risp = 1/np.polyval([1/(2*np.pi*x[0])**2,1/(2*np.pi*x[0]*x[1]),1],s)*np.exp(-s*x[2])
    return risp
                        
def loss(x,f,H):
    """ Loss function for TF fit.
    Parameters
    ----------
    x : 3-shaped array float [Hz,-,s]
        TF parameters [fc,Q,T]
    f : N-shaped array float [Hz]
        Frequencies at which FRF is measured
    H : N-shaped array float [-]
        Measured FRF values at frequencies f
    Returns
    ----------
    l :     float [-] 
            Measured loss between TF and FRF 
    """
    risp = Lowpass(x,f)
    l = np.linalg.norm((risp-H).reshape(-1, 1), axis=1).sum()  
    return l                       
        
def estimateLowpass(f,H,x0,options={'xatol':1e-4,'disp': True}):
    """ Loss function for TF fit.
    Parameters
    ----------
    f : N-shaped array float [Hz]
        Frequencies at which FRF is measured
    H : N-shaped array float [-]
        Measured FRF values at frequencies f
    x : 3-shaped array float [Hz,-,s]
        Initial guess for fit [fc,Q,T]
    options :   Dictionary
                Dictionary of solver options for scipy.minimize function
    Returns
    ----------
    res.x : 3-shaped array float [Hz,-,s]    
            Fitted TF parameters [fc,Q,T]
    """
    pass_to_loss = lambda x: loss(x,f,H)
    res = minimize(pass_to_loss,x0,method='nelder-mead',options=options)
    return res.x
    
def TFDelay(x,f,num,den):
    """ 2nd order lowpass with delay TF.
    Parameters
    ----------
    x : float [s]
        TF parameters [delay]
    f : N-shaped array float [Hz]
        Frequencies at which TF is evaluated
    Returns
    ----------
    risp :  N-shaped array float [-]
            TF values evaluated at frequencies f 
    """
    s = 1j*f*2*np.pi
    risp = np.polyval(num,s)/np.polyval(den,s)*np.exp(-s*x)
    return risp
                        
def Delayloss(x,f,H,num,den):
    """ Loss function for TF fit.
    Parameters
    ----------
    x : 3-shaped array float [-,-,s]
        TF parameters [num,den,T]
    f : N-shaped array float [Hz]
        Frequencies at which FRF is measured
    H : N-shaped array float [-]
        Measured FRF values at frequencies f
    Returns
    ----------
    l :     float [-] 
            Measured loss between TF and FRF 
    """
    risp = TFDelay(x,f,num,den)
    l = np.linalg.norm((risp-H).reshape(-1, 1), axis=1).sum()  
    return l                       
        
def Delayestimate(f,H,x0,num,den,options={'xatol':1e-4,'disp': True}):
    """ Loss function for TF fit.
    Parameters
    ----------
    f : N-shaped array float [Hz]
        Frequencies at which FRF is measured
    H : N-shaped array float [-]
        Measured FRF values at frequencies f
    x : float [s]
        Initial guess for delay fit
    options :   Dictionary
                Dictionary of solver options for scipy.minimize function
    Returns
    ----------
    res.x : 3-shaped array float [Hz,-,s]    
            Fitted TF parameters [fc,Q,T]
    """
    pass_to_loss = lambda x: Delayloss(x,f,H,num,den)
    res = minimize(pass_to_loss,x0,method='nelder-mead',options=options)
    return res.x
