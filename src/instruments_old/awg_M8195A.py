import numpy as np
import math

from .basic_instrument import BasicInstrument
from pyarbtools.instruments import M8195A
from pyarbtools.wfmBuilder import sine_generator

def  farey_fraction(x, max_iter=1000, end_tol=1e-4):
    '''x is the irrational float to approximate
    between a/b and c/d, where a,b,c and d are ints'''
    a = math.floor(x)
    b = 1
    c = math.ceil(x)
    d = 1
    for iter in range(max_iter):
        new_nom = a+c
        new_denom = b+d
        new_approx = new_nom/new_denom
        if new_approx<=x:
            a = new_nom
            b = new_denom
        else:
            c = new_nom
            d = new_denom
        if np.abs(new_approx-x)<=end_tol:
            #always return the best approx as the first two ints.
            best_approx = np.argmin((np.abs(a/b-x), np.abs(c/d-x)))
            if best_approx==0:
                return (a, b, c, d)
            else:
                return (c, d, a, b)
    raise RecursionError("Could not find good enough approximation."
                         "Increase max_iter."
                         f"Current approx: {(a, b, c, d)}")
        

class AWG_M8195A(BasicInstrument, M8195A):
    """Customized object interfacing to PyArbTools.
    Main goal is to avoid misconfiguration. Also standardizes
    configuration saving.
    """
    
    #default config
    #NOTE: pyarbtools.instruments.M8195A.configure() (checked against the
    #pip-installable pyarbtools 2025.06.1) only accepts dacMode, memDiv,
    #fs, refSrc, refFreq, amp1/2/3/4 and func - it raises KeyError on
    #anything else. out_refSrc/mem_mode1/mem_mode4 used to be listed here
    #and made this constructor crash immediately; if your installed
    #pyarbtools is an older/different version that does support them,
    #add them back.
    default_config = {
        'dacMode':'dual',
        'memDiv':1,
        'fs':65e9,
        'refSrc':'ext',
        'refFreq':10e6,
        'amp1':1,
        'amp4':1,
        'func':'arb'
    }
    
    def __init__(self, visa_address, nickname, config=None):
        BasicInstrument.__init__(self, visa_address, nickname, config)
        M8195A.__init__(self, visa_address,
                        apiType="pyvisa",
                        protocol="vxi11",
                        port=0,
                        reset=True)
        #Reset already done
        #Call PyArbTools configure() method.
        self.configure(**self.config)
        self.min_rate = 53.76e9
        self.max_rate = 65e9
    
    def find_waveform_k_nper(self, freq, max_iterations=1000):
        k = 1 #number of granularities of data
        n_per = 1 #number of periods of sine
        for id_iter in range(max_iterations):
            #round to kSa/s (limit of AWG)
            current_rate = 1000*round(self.gran*freq*k/n_per/1000)
            if current_rate>=self.max_rate:
                n_per+=1
            elif current_rate<=self.min_rate:
                k+=1
                n_per = k+1
            else:
                return k, n_per, current_rate
        #if it exits the loop, it has failed to find params
        raise RuntimeError("Couldn't find k and n_per for"
                               f"freq {freq}Hz.")        
        
    def send_sine(self, freq, phase, channel, amp=None):
        '''
        Generates points for a sine at freq and phase (radians),
        calculating and changing the rate for good granularity
        requirement. Changes amplitude of the channel (if not None)
        and sends signal.
        '''
        k, n_per, rate = self.find_waveform_k_nper(freq)
        #time points, actual time caculated with sample rate
        t_points = np.arange(k*self.gran)
        #period in number of points
        period = k*self.gran/n_per
        amp_points = np.sin(2*np.pi*t_points/period + phase)
        self.configure(fs=rate)
        if amp is not None:
            self.configure(**{f'amp{channel}':amp})
        seg_id = self.download_wfm(amp_points, ch=channel, name="wfm")
        self.play(seg_id, ch=channel)
        return seg_id
    
    def send_sine_keep_rate(self, freq, phase, channel, amp=None):
        '''
        Generates points for a sine at freq and phase (radians),
        changes amplitude of the channel (if not None)
        and sends signal.
        '''
        k, n_per, rate = self.find_waveform_k_nper(freq)
        if rate!=self.fs:
            raise RuntimeError("Tried to send additional signal"
                               " incompatible with current rate."
                               f"Current {self.fs}Hz, needed {rate}Hz.")
        #time points, actual time caculated with sample rate
        t_points = np.arange(k*self.gran)
        #period in number of points
        period = k*self.gran/n_per
        amp_points = np.sin(2*np.pi*t_points/period + phase)
        if amp is not None:
            self.configure(**{f'amp{channel}':amp})
        seg_id = self.download_wfm(amp_points, ch=channel, name="wfm")
        self.play(seg_id, ch=channel)
        return seg_id
    
    def send_sine_force_keep_rate(self, freq, phase, channel, amp=None):
        '''
        Generates points for a sine at freq and phase (radians),
        changes amplitude of the channel (if not None)
        and sends signal. Will make the sample as long as necessary to
        keep the sample rate.
        '''
        decimal_val = self.gran*freq/self.fs
        #Farey fraction
        #max_iter*self.gran should be just below memory max sample size
        #1e-6 end_tol should make it exact.
        n_per, k, _, _ = farey_fraction(decimal_val, max_iter=1999, end_tol=1e-6)
        #time points, actual time calculated with sample rate
        t_points = np.arange(k*self.gran)
        #period in number of points
        period = k*self.gran/n_per
        amp_points = np.sin(2*np.pi*t_points/period + phase)
        if amp is not None:
            self.configure(**{f'amp{channel}':amp})
        seg_id = self.download_wfm(amp_points, ch=channel, name="wfm")
        self.play(seg_id, ch=channel)
        return seg_id