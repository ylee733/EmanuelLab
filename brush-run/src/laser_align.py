import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

settings = {
    'xMirror_output': 'Dev2/ao0',
    'yMirror_output': 'Dev2/ao1',
    'laser_output': '/Dev2/port0/line5',

    'Fs': 20000,  # in samples/s 
    
    'mm_X': 5, # in mm
    'mm_Y': 5, # in mm
    
    'V_per_mm': 1/6.25, # V/mm
    'duration': 5, # in s
    'minV' : -10, # in Volts. Minimum voltage for mirror
    'maxV' : 10, # in Volts. Maximum voltage for mirror
    'lasFreq': 40, # in Hz
    'lasDur': 0.00015 # in s (30 microS)

# 0.0003 for finding center 01/08/2025

#9,3 is 0,0 for test on  12/14/23
#12.5mm/2V
## 0.007

}
def setupTasks(settings):
    # calculate the trial duration needed to sample every region and numSamples

    numSamples = settings['duration'] * settings['Fs']
  
    ao_task = nidaqmx.Task()
    ao_task.ao_channels.add_ao_voltage_chan(settings['xMirror_output'], name_to_assign_to_channel='x_out')
    ao_task.ao_channels.add_ao_voltage_chan(settings['yMirror_output'], name_to_assign_to_channel='y_out')
    ao_task.timing.cfg_samp_clk_timing(settings['Fs'], samps_per_chan=numSamples)

    do_task = nidaqmx.Task()
    do_task.do_channels.add_do_chan(settings['laser_output'], name_to_assign_to_lines='laser')
    do_task.timing.cfg_samp_clk_timing(settings['Fs'], samps_per_chan=numSamples)

    return ao_task, do_task

def align(ao_task, do_task, settings):
    ### Generate alignment pulse based on duration and voltages in settings ###

    V_X = settings['mm_X'] * settings['V_per_mm']
    V_Y = settings['mm_Y'] * settings['V_per_mm']
    ## Construct stimulus
    print(V_X,V_Y)
    t = np.arange(0,settings['duration'],1/settings['Fs'])
    laserStarts_s = np.arange(1,settings['duration'],1/settings['lasFreq'])
    laserStarts_samples = np.int32(laserStarts_s * settings['Fs'])
    laserEnds_samples = np.int32(laserStarts_samples + settings['lasDur']*settings['Fs'])
    lz1 = np.zeros(settings['Fs'] * settings['duration'],dtype=bool)
    for start, end in zip(laserStarts_samples,laserEnds_samples):
        # print('sample duration of laser pulse',end - start)
        lz1[start:end] = True
    

    x1 = np.zeros(len(lz1))
    y1 = np.zeros(len(lz1))
    x1[:] = V_X
    y1[:] = V_Y
  
    if np.abs(V_X) > 10:
        print('X voltage out of bounds')
        return -1
    elif np.abs(V_X) > 10:
        print('Y voltage out of bounds')
        return-1
    lz1[-1] = 0

    ao_out = np.zeros((2,len(lz1)))
    ao_out[0,:] = x1
    ao_out[1,:] = y1
    do_out = lz1
  
    ## writing daq outputs onto device
    do_task.write(do_out,timeout=10+settings['duration'])
    ao_task.write(ao_out,timeout=10+settings['duration'])
    do_task.start()
    ao_task.start()

    do_task.wait_until_done(timeout=10+settings['duration'])
    

    ## stopping tasks
    do_task.stop()
    ao_task.stop()
  
    do_task.close()
    ao_task.close()
  
    ao_data = ao_out
    do_data = do_out

    return ao_data, do_data

ao_task, do_task = setupTasks(settings)
ao_data, do_data = align(ao_task, do_task, settings)
print("Task complete.")


# plt.plot(do_data)
# plt.show()
# plt.close()
