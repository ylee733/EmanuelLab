import nidaqmx
import numpy as np
import scipy.signal
import time
import compress_pickle as pickle
from json import load as jsonload, dump as jsondump
import os
import FreeSimpleGUI as sg
import threading
import scipy.stats  
import matplotlib.pyplot as plt  

# ---------------- Global definitions ----------------
SETTINGS_FILE = os.path.join(os.getcwd(), r'settings_file.cfg')
DEFAULT_SETTINGS = {
    'trigger_input': '/Dev3/PFI0',
    'trial_start': '/Dev3/port0/line0',
    'dir1': '/Dev3/port0/line1',
    'dir2': '/Dev3/port0/line2',
    'reward_output': '/Dev3/port0/line3',
    '2p_trigger': '/Dev3/port0/line4',
    'laser': '/Dev3/port0/line5',
    'lick_input': '/Dev3/port0/line7',
    'x_output': '/Dev3/ao0',
    'y_output': '/Dev3/ao1'
}

SETTINGS_KEYS_TO_ELEMENT_KEYS = {
    'trigger_input': '-TRIGGER INPUT-',
    'trial_start': '-TRIAL START-',
    'dir1': '-DIR 1-',
    'dir2': '-DIR 2-',
    'reward_output': '-REWARD OUT-',
    '2p_trigger': '-2P TRIGGER-',
    'laser': '-LASER-',
    'lick_input': '-LICK IN-',
    'x_output': '-X OUT-',
    'y_output': '-Y OUT-'
}

# Global pause and end-session events.
end_session_event = threading.Event()
pause_event = threading.Event()
pause_event.set()  # start in running state

##################### Load/Save Settings File #####################
def load_settings(settings_file, default_settings):
    try:
        with open(settings_file, 'r') as f:
            settings = jsonload(f)
    except Exception as e:
        sg.popup_quick_message(f'exception {e}', 
                               'No settings file found... will create one for you', 
                               keep_on_top=True, background_color='red', text_color='white')
        settings = default_settings
        save_settings(settings_file, settings, None)
    return settings

def save_settings(settings_file, settings, values):
    if values:
        for key in SETTINGS_KEYS_TO_ELEMENT_KEYS:
            try:
                settings[key] = values[SETTINGS_KEYS_TO_ELEMENT_KEYS[key]]
            except Exception as e:
                print(f'Problem updating settings from window values. Key = {key}')
    with open(settings_file, 'w') as f:
        jsondump(settings, f)
    sg.popup('Settings saved')

##################### Create Settings Window #####################
def create_settings_window(settings):
    sg.theme('Default1')
    def TextLabel(text): 
        return sg.Text(text+':', justification='r', size=(15,1))
    layout = [
        [sg.Text('DAQ Settings', font='Any 15')],
        [TextLabel('trigger_input'), sg.Input(key='-TRIGGER INPUT-')],
        [TextLabel('trial start'), sg.Input(key='-TRIAL START-')],
        [TextLabel('dir1'), sg.Input(key='-DIR 1-')],
        [TextLabel('dir2'), sg.Input(key='-DIR 2-')],
        [TextLabel('reward_output'), sg.Input(key='-REWARD OUT-')],
        [TextLabel('2p_trigger'), sg.Input(key='-2P TRIGGER-')],
        [TextLabel('laser'), sg.Input(key='-LASER-')],
        [TextLabel('lick_input'), sg.Input(key='-LICK IN-')],
        [TextLabel('x_output'), sg.Input(key='-X OUT-')],
        [TextLabel('y_output'), sg.Input(key='-Y OUT-')],
        [sg.Button('Save'), sg.Button('Exit')]
    ]
    window = sg.Window('Settings', layout, keep_on_top=True, finalize=True)
    for key in SETTINGS_KEYS_TO_ELEMENT_KEYS:
        try:
            window[SETTINGS_KEYS_TO_ELEMENT_KEYS[key]].update(value=settings[key])
        except Exception as e:
            print(f'Problem updating PySimpleGUI window from settings. Key = {key}')
    return window

##################### Create Laser Settings Window #####################
def createLaserSettingsWindow():
    sg.theme('Default1')
    layout = [
        [sg.Text('Enter Laser Parameters', font='Any 15')],
        [sg.Text('DC X (mm):', size=(15,1)), sg.Input(default_text='4', key='-DCX-')],
        [sg.Text('DC Y (mm):', size=(15,1)), sg.Input(default_text='-4', key='-DCY-')],
        [sg.Text('FS1 X (mm):', size=(15,1)), sg.Input(default_text='0.45', key='-FS1X-')],
        [sg.Text('FS1 Y (mm):', size=(15,1)), sg.Input(default_text='0.4', key='-FS1Y-')],
        [sg.Text('HS1 X (mm):', size=(15,1)), sg.Input(default_text='0', key='-HS1X-')],
        [sg.Text('HS1 Y (mm):', size=(15,1)), sg.Input(default_text='-0.2', key='-HS1Y-')],
        [sg.Text('PPC X (mm):', size=(15,1)), sg.Input(default_text='-0.15', key='-PPCX-')],
        [sg.Text('PPC Y (mm):', size=(15,1)), sg.Input(default_text='-1.3', key='-PPCY-')],

        [sg.Text('HS1 probability:', size=(15,1)), sg.Input(default_text='0.1', key='-HS1X-')],
        [sg.Text('FS1 probability', size=(15,1)), sg.Input(default_text='0.1', key='-HS1Y-')],
        [sg.Text('PPC probability:', size=(15,1)), sg.Input(default_text='0.1', key='-PPCX-')],
        [sg.Text('DC probability', size=(15,1)), sg.Input(default_text='0.7', key='-PPCY-')],

        [sg.Text('X Reference (mm):', size=(15,1)), sg.Input(default_text='23.0', key='-REFX-')],
        [sg.Text('Y Reference (mm):', size=(15,1)), sg.Input(default_text='-26.0', key='-REFY-')],
        [sg.Text('V/mm Conversion:', size=(15,1)), sg.Input(default_text='0.16', key='-Conversion-')],
        [sg.Text('Laser Start (s):', size=(15,1)), sg.Input(default_text='0.9', key='-LaserStart-')],
        [sg.Text('Laser End (s):', size=(15,1)), sg.Input(default_text='2.1', key='-LaserEnd-')],
        [sg.Text('Laser Frequency (Hz):', size=(15,1)), sg.Input(default_text='40', key='-LaserFreq-')],
        [sg.Text('Laser Pulse (ms):', size=(15,1)), sg.Input(default_text='0.5', key='-LaserPW-')],
        [sg.Button('Save'), sg.Button('Cancel')]
    ]
    window = sg.Window('Laser Settings', layout, modal=True)
    event, values = window.read()
    window.close()
    if event == 'Save':
        return values
    else:
        return None

##################### Set up DAQ tasks #####################
def setupDaq(settings, taskParameters, setup='task'):
    numSamples = int(taskParameters['Fs'] * taskParameters['trialDuration'])
    if setup == 'task':
        di_task = nidaqmx.Task()
        di_task.di_channels.add_di_chan(settings['lick_input'], name_to_assign_to_lines='lick')
        di_task.timing.cfg_samp_clk_timing(taskParameters['Fs'], source="OnboardClock",
                                            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        ao_task = None
        if taskParameters.get('laserOutput', False):
            ao_task = nidaqmx.Task()
            ao_task.ao_channels.add_ao_voltage_chan(settings['x_output'], name_to_assign_to_channel='x_out')
            ao_task.ao_channels.add_ao_voltage_chan(settings['y_output'], name_to_assign_to_channel='y_out')
            ao_task.timing.cfg_samp_clk_timing(taskParameters['Fs'], samps_per_chan=numSamples)
            ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(settings['trigger_input'])
        do_task = nidaqmx.Task()
        do_task.do_channels.add_do_chan(settings['trial_start'], name_to_assign_to_lines='trial_start')
        do_task.do_channels.add_do_chan(settings['dir1'], name_to_assign_to_lines='dir1')
        do_task.do_channels.add_do_chan(settings['dir2'], name_to_assign_to_lines='dir2')
        do_task.do_channels.add_do_chan(settings['reward_output'], name_to_assign_to_lines='reward_output')
        do_task.do_channels.add_do_chan(settings['2p_trigger'], name_to_assign_to_lines='2P_trigger')
        # For operant sessions with laser, reserve an extra digital channel for laser pulses.
        if taskParameters.get('laserOutput', False):
            do_task.do_channels.add_do_chan(settings['laser'], name_to_assign_to_lines='laser')
        do_task.timing.cfg_samp_clk_timing(taskParameters['Fs'], source="OnboardClock",
                                            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                            samps_per_chan=numSamples)
        return di_task, ao_task, do_task, setup

##################### Classical runTrial Function #####################
lastTrialGo = False
def runTrial(di_task, do_task, taskParameters, settings, trialIndex, totalTrials):
    # Classical trial (without laser AO data)
    numSamples = int(taskParameters['Fs'] * taskParameters['trialDuration'])
    stimTime_samples = int(taskParameters['stimTime'] * taskParameters['Fs'])
    stimDuration_samples = int(taskParameters['stimDuration'] * taskParameters['Fs'])
    samplesToRewardStart = int(taskParameters['rewardWindowOnset'] * taskParameters['Fs'])
    samplesToRewardEnd = int(samplesToRewardStart + taskParameters['rewardWindowDuration'] * taskParameters['Fs'])
    pulse_ms = 10
    gap_ms = 500
    do_out = np.zeros([5, numSamples], dtype='bool')
    do_out[0, 1:-1] = True
    pulseWidth = int(0.001 * pulse_ms * taskParameters['Fs'])
    gapWidth = int(0.001 * gap_ms * taskParameters['Fs'])

    do_out[4, 5:pulseWidth] = True
    # start2 = -5 - pulseWidth
    # do_out[4, start2:-5] = True

    goTrial = np.random.binomial(1, taskParameters['goProbability'])
    global lastTrialGo
    if taskParameters.get('alternate', False):
        goTrial = not lastTrialGo
    if goTrial:
        if taskParameters['sessionMode'] == 'clockwise':
            do_out[2, stimTime_samples:stimTime_samples + stimDuration_samples] = True
        elif taskParameters['sessionMode'] == 'counterclockwise':
            do_out[1, stimTime_samples:stimTime_samples + stimDuration_samples] = True
        else:
            raise ValueError("Invalid session mode in taskParameters")
        reward_pulse_dur = int(0.04 * taskParameters['Fs'])
        start_pulse = samplesToRewardEnd - reward_pulse_dur
        if start_pulse < samplesToRewardStart:
            start_pulse = samplesToRewardStart
        do_out[3, start_pulse:samplesToRewardEnd] = True
        print("classical go trial")
    else:
        if taskParameters['sessionMode'] == 'clockwise':
            do_out[1, stimTime_samples:stimTime_samples + stimDuration_samples] = True
        elif taskParameters['sessionMode'] == 'counterclockwise':
            do_out[2, stimTime_samples:stimTime_samples + stimDuration_samples] = True
        else:
            raise ValueError("Invalid session mode in taskParameters")
        print("classical no-go trial")
    pretrial_licks = {}

    print("pre-trial lick monitoring start")
    lick_free_duration = taskParameters['lickFree']
    chunk_duration = 0.1
    chunk_samples = int(taskParameters['Fs'] * chunk_duration)
    detect_lick = True
    pretrial_licks[trialIndex] = []

    while detect_lick:
        di_task.timing.cfg_samp_clk_timing(taskParameters['Fs'],
                                        sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        try:
            di_task.start()
            start_time = time.time()
            has_licks = False

            while (time.time() - start_time) < lick_free_duration and not has_licks:
                data = di_task.read(number_of_samples_per_channel=chunk_samples, timeout=chunk_duration+0.1)
                if np.any(data):
                    has_licks = True
                    print("lick detected, restarting monitoring")

                    # Store lick data in pretrial_licks
                    timestamps = np.arange(start_time, start_time + chunk_duration, 1 / taskParameters['Fs'])[:len(data)]
                    lick_times = timestamps[np.array(data, dtype=bool)]
                    pretrial_licks[trialIndex].extend(lick_times.tolist())

                time.sleep(0.01)

            if not has_licks:
                print("no licks detected, proceeding with trial")
                detect_lick = False

        except Exception as e:
            print(f"Error during lick detection: {e}")

        finally:
            di_task.stop()

        if detect_lick:
            time.sleep(0.5)

    # Proceed with trial after lick-free period
    di_task.timing.cfg_samp_clk_timing(taskParameters['Fs'],
                                    sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                    samps_per_chan=numSamples)

    do_task.write(do_out)
    di_task.start()
    do_task.start()
    try:
        di_data = []
        chunk_size = int(taskParameters['Fs'] * 0.1)
        samples_read = 0
        while samples_read < numSamples:
            samples_to_read = min(chunk_size, numSamples - samples_read)
            chunk = di_task.read(number_of_samples_per_channel=samples_to_read, timeout=1.0)
            di_data.extend(chunk)
            samples_read += len(chunk)
            time.sleep(0.01)
        di_data = np.array(di_data)
        do_task.wait_until_done(timeout=taskParameters['trialDuration']+1)
    finally:
        do_task.stop()
        di_task.stop()
    if goTrial:
        if np.sum(di_data[samplesToRewardStart:samplesToRewardEnd]) > 0:
            result = 'hit'
            print('\tHit')
        else:
            result = 'miss'
            print('\tMiss')
        lastTrialGo = True
    else:
        if np.sum(di_data[samplesToRewardStart:samplesToRewardEnd]) > 0:
            result = 'FA'
            print('\tFalse Alarm')
        else:
            result = 'CR'
            print('\tCorrect Rejection')
        lastTrialGo = False
    return di_data, do_out, result, pretrial_licks[trialIndex]

##################### Operant runTrial Function #####################
def runOperantTrial(di_task, ao_task, do_task, taskParameters, settings, trialIndex, totalTrials, probe=False):
    # Timing calculations.
    numSamples = int(taskParameters['Fs'] * taskParameters['trialDuration'])
    stimTime_samples = int(taskParameters['stimTime'] * taskParameters['Fs'])
    stimDuration_samples = int(taskParameters['stimDuration'] * taskParameters['Fs'])
    samplesToRewardStart = int(taskParameters['rewardWindowOnset'] * taskParameters['Fs'])
    samplesToRewardEnd = int(samplesToRewardStart + taskParameters['rewardWindowDuration'] * taskParameters['Fs'])
    pulse_ms = 20
    gap_ms = 750
    # Allocate AO buffer and extend DO channels (6 channels: extra channel for laser digital).
    ao_out = np.zeros([2, numSamples])
    num_do_channels = 5 if not taskParameters.get('laserOutput', False) else 6
    do_out = np.zeros([num_do_channels, numSamples], dtype='bool')

    do_out[0, 1:-1] = True
    pulseWidth = int(0.001 * pulse_ms * taskParameters['Fs'])
    gapWidth = int(0.001 * gap_ms * taskParameters['Fs'])

    do_out[4, 5:pulseWidth] = True
    # start2 = -5 - pulseWidth
    # do_out[4, start2:-5] = True


    goTrial = np.random.binomial(1, taskParameters['goProbability'])
    if goTrial:
        if taskParameters['sessionMode'] == 'clockwise':
            do_out[2, stimTime_samples:stimTime_samples+stimDuration_samples] = True
            # reward delievered not in probe
            if not probe:
                do_out[3, samplesToRewardStart:samplesToRewardEnd] = True
        elif taskParameters['sessionMode'] == 'counterclockwise':
            do_out[1, stimTime_samples:stimTime_samples+stimDuration_samples] = True
            # reward delievered not in probe
            if not probe:
                do_out[3, samplesToRewardStart:samplesToRewardEnd] = True
        else:
            raise ValueError("Invalid session mode in taskParameters")
        print("operant go trial")
    else:
        if taskParameters['sessionMode'] == 'clockwise':
            do_out[1, stimTime_samples:stimTime_samples+stimDuration_samples] = True
        elif taskParameters['sessionMode'] == 'counterclockwise':
            do_out[2, stimTime_samples:stimTime_samples+stimDuration_samples] = True
        else:
            raise ValueError("Invalid session mode in taskParameters")
        print("operant no-go trial")
    # --- Laser functionality for operant session ---
    if taskParameters.get('laserOutput', False):
        # laser on for 90% of non probe trials
        laserOn = (not probe) and (np.random.rand() < 0.9)
        if laserOn:
            laserLoc = 'FS1'
            # FS1 coordinates
            x = (taskParameters['x_FS1'] + taskParameters['x_Ref']) * taskParameters['V_div_mm']
            y = (taskParameters['y_FS1'] + taskParameters['y_Ref']) * taskParameters['V_div_mm']
            ao_out[0, 1:-1] = x
            ao_out[1, 1:-1] = y

            laserStart_s = np.arange(taskParameters['laserStart'], taskParameters['laserEnd'], 1/taskParameters['laserFrequency'])
            laserEnd_s = laserStart_s + taskParameters['laserPulseLength'] * 0.001
            laserStart_samples = np.int32(laserStart_s * taskParameters['Fs'])
            laserEnd_samples = np.int32(laserEnd_s * taskParameters['Fs'])
            for start_sample, end_sample in zip(laserStart_samples, laserEnd_samples):
                do_out[5, start_sample:end_sample] = True
        else:
            laserLoc = 'null'
    else:
        laserLoc = 'null'
    # --- End Laser functionality ---
    pretrial_licks = {}

    print("pre-trial lick monitoring start")
    lick_free_duration = taskParameters['lickFree']
    chunk_duration = 0.1
    chunk_samples = int(taskParameters['Fs'] * chunk_duration)
    detect_lick = True
    pretrial_licks[trialIndex] = []

    while detect_lick:
        di_task.timing.cfg_samp_clk_timing(taskParameters['Fs'],
                                        sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        try:
            di_task.start()
            start_time = time.time()
            has_licks = False

            while (time.time() - start_time) < lick_free_duration and not has_licks:
                data = di_task.read(number_of_samples_per_channel=chunk_samples, timeout=chunk_duration+0.1)
                if np.any(data):
                    has_licks = True
                    print("lick detected, restarting monitoring")

                    # Store lick data in pretrial_licks
                    timestamps = np.arange(start_time, start_time + chunk_duration, 1 / taskParameters['Fs'])[:len(data)]
                    lick_times = timestamps[np.array(data, dtype=bool)]
                    pretrial_licks[trialIndex].extend(lick_times.tolist())

                time.sleep(0.01)

            if not has_licks:
                print("no licks detected, proceeding with trial")
                detect_lick = False

        except Exception as e:
            print(f"Error during lick detection: {e}")

        finally:
            di_task.stop()

        if detect_lick:
            time.sleep(0.5)

    # Proceed with trial after lick-free period
    di_task.timing.cfg_samp_clk_timing(taskParameters['Fs'],
                                    sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                    samps_per_chan=numSamples)

    do_task.write(do_out)
    if taskParameters.get('laserOutput', False) and ao_task is not None:
        ao_task.write(ao_out)
    di_task.start()
    if taskParameters.get('laserOutput', False) and ao_task is not None:
        ao_task.start()
    do_task.start()
    do_task.wait_until_done(timeout=taskParameters['trialDuration']+1)
    try:
        di_data = []
        chunk_size = int(taskParameters['Fs'] * 0.1)
        samples_read = 0
        while samples_read < numSamples:
            samples_to_read = min(chunk_size, numSamples - samples_read)
            chunk = di_task.read(number_of_samples_per_channel=samples_to_read, timeout=1.0)
            di_data.extend(chunk)
            samples_read += len(chunk)
            time.sleep(0.01)
        di_data = np.array(di_data)
        do_task.wait_until_done(timeout=taskParameters['trialDuration']+1)
    finally:
        do_task.stop()
        if taskParameters.get('laserOutput', False) and ao_task is not None:
            ao_task.stop()
        di_task.stop()
    if goTrial:
        if np.sum(di_data[samplesToRewardStart:samplesToRewardEnd]) > 0:
            result = 'hit'
            print("\tHit")
        else:
            result = 'miss'
            print("\tMiss")
    else:
        if np.sum(di_data[samplesToRewardStart:samplesToRewardEnd]) > 0:
            result = 'FA'
            print("\tFalse Alarm")
        else:
            result = 'CR'
            print("\tCorrect Rejection")
    return di_data, ao_out, do_out, result, laserLoc,  pretrial_licks[trialIndex]

##################### runTask (merged) #####################
def runTask(di_task, ao_task, do_task, taskParameters, settings):
    di_data = {}
    do_data = {}
    ao_data = {}
    pretrial_licks = {}
    results = []
    laserLocs = []
    originalProb = taskParameters['goProbability']
    if taskParameters['save']:
        fileName = os.path.join(taskParameters['savePath'],
                                f"{time.strftime('%Y%m%d_%H%M%S')}_{taskParameters['animal']}.gz")
    for trial in range(taskParameters['numTrials']):
        pretrial_licks[trial] = []
        while not pause_event.is_set():
            time.sleep(0.1)
        if end_session_event.is_set():
            print("End Session button pressed. Ending session early.")
            break
        print(f'On trial {trial+1} of {taskParameters["numTrials"]}')
        if taskParameters['conditioningMode'] == "Operant":
            # decide if this trial is in the probe window
            is_probe = (trial >= taskParameters['startProbes']
                        and trial <  taskParameters['endProbes'])
            di_data[trial], ao_data[trial], do_data[trial], result, laserLoc, pretrial_licks[trial] = runOperantTrial(
                di_task, ao_task, do_task,
                taskParameters, settings,
                trial, taskParameters['numTrials'],
                probe=is_probe)
            laserLocs.append(laserLoc)
        else:
            di_data[trial], do_data[trial], result, pretrial_licks[trial] = runTrial(di_task, do_task, taskParameters, settings, trial, taskParameters['numTrials'])
        results.append(result)
        temp = np.array(results)
        try:
            hitRate = np.sum(temp=='hit')/(np.sum(temp=='hit')+np.sum(temp=='miss')+1)
            FARate = np.sum(temp=='FA')/(np.sum(temp=='FA')+np.sum(temp=='CR')+1)
            print('\tHit Rate = {0:0.2f}, FA Rate = {1:0.2f}, d\' = {2:0.2f}'.format(hitRate, FARate, dprime(hitRate, FARate)))
        except ZeroDivisionError:
            pass
        if result == 'FA':
            time.sleep(taskParameters['falseAlarmTimeout'])
        else: 
            time.sleep(1)
        last20 = temp[-20:]
        last10 = temp[-10:]
        if len(last20) > 0:
            FA_den = np.sum(np.isin(last20, ['FA','CR']))
            hit_den = np.sum(np.isin(last20, ['hit','miss']))
            FA_rate_last20 = np.sum(last20=='FA')/FA_den if FA_den else 0
            hitRate_last20 = np.sum(last20=='hit')/hit_den if hit_den else 0
            print('\tHit Rate Last 20 = {:.2f}; Total hits = {}'.format(hitRate_last20, np.sum(temp=='hit')))
        # if len(last20) == 20 and np.all(np.isin(last20, ['miss', 'CR'])):
        #     print('\n\n Task aborted due to 20 consecutive misses')
        #     abortTask = True
        #     break
        if taskParameters.get('sculptBehavior', False):
            if len(last20) == 20 and FA_rate_last20 > 0.9:
                taskParameters['goProbability'] = 0
                print('\t\tForced no-go trial due to high FA rate in last 20 trials')
            else:
                taskParameters['goProbability'] = originalProb
        if taskParameters['save'] and trial % 5 == 0:
            outDict = {
                'taskParameters': taskParameters,
                'di_data': {**di_data},
                'di_channels': di_task.channel_names,
                'do_data': {**do_data},
                'do_channels': do_task.channel_names,
                'ao_data': {**ao_data},
                'ao_channels': ao_task.channel_names if ao_task is not None else 'None',
                'results': np.array(results),
                'laserLocs': np.array(laserLocs),
                'pretrial_licks': pretrial_licks 
            }
            pickle.dump(outDict, fileName)
    print('\n\nTask Finished, {} rewards delivered\n'.format(np.sum(temp=='hit')))
    print('\tHits: {}'.format(np.sum(temp=='hit')))
    print('\tMisses: {}'.format(np.sum(temp=='miss')))
    print('\tFalse Alarms: {}'.format(np.sum(temp=='FA')))
    print('\tCorrect Rejections: {}'.format(np.sum(temp=='CR')))
    print('\tAmount of water consumed: {} mL'.format(np.sum(temp=='hit') * taskParameters['dispensedVolume']))
    taskParameters['goProbability'] = originalProb
    if taskParameters['save']:
        print('...saving final data...\n')
        outDict = {
            'taskParameters': taskParameters,
            'di_data': {**di_data},
            'di_channels': di_task.channel_names,
            'do_data': {**do_data},
            'do_channels': do_task.channel_names,
            'ao_data': {**ao_data},
            'ao_channels': ao_task.channel_names if ao_task is not None else 'None',
            'results': np.array(results),
            'laserLocs': np.array(laserLocs),
            'pretrial_licks': pretrial_licks 
        }
        pickle.dump(outDict, fileName)
        print('Data saved in {}\n'.format(fileName))

def dprime(hitRate, falseAlarmRate):
    return scipy.stats.norm.ppf(hitRate) - scipy.stats.norm.ppf(falseAlarmRate)

##################### Update Parameters #####################
def updateParameters(values):
    taskParameters = {}
    taskParameters['startProbes'] = int(values['-StartProbes-'])
    taskParameters['endProbes'] = int(values['-EndProbes-'])
    taskParameters['numTrials'] = int(values['-NumTrials-'])
    taskParameters['Fs'] = int(values['-SampleRate-'])
    taskParameters['downSample'] = values['-DownSample-']
    taskParameters['trialDuration'] = float(values['-TrialDuration-'])
    taskParameters['falseAlarmTimeout'] = float(values['-FalseAlarmTimeout-'])
    taskParameters['goProbability'] = float(values['-GoProbability-'])
    taskParameters['stimTime'] = float(values['-StimTime-'])
    taskParameters['stimDuration'] = float(values['-StimDuration-'])
    taskParameters['rewardWindowOnset'] = float(values['-RewardWindowOnset-'])
    taskParameters['rewardWindowDuration'] = float(values['-RewardWindowDuration-'])
    taskParameters['savePath'] = values['-SavePath-']
    taskParameters['save'] = values['-Save-']
    taskParameters['animal'] = values['-Animal-']
    taskParameters['lickFree'] = float(values['-LickFree-'])
    taskParameters['dispensedVolume'] = float(values['-DispensedVolume-'])
    taskParameters['sculptBehavior'] = values['-SculptBehavior-']
    # Laser option: if user checks "Use Laser & XY"
    taskParameters['laserOutput'] = values.get('-Laser-', False)

    taskParameters['V_div_mm'] = float(values.get('-Conversion-', 0.16))
    taskParameters['x_Ref'] = float(values.get('-REFX-', 0))
    taskParameters['y_Ref'] = float(values.get('-REFY-', 0))
    taskParameters['laserFrequency'] = float(values.get('-LaserFreq-', 40))
    taskParameters['laserPulseLength'] = float(values.get('-LaserPW-', 0.5))
    taskParameters['laserStart'] = float(values.get('-LaserStart-', 0.9))
    taskParameters['laserEnd'] = float(values.get('-LaserEnd-', 2.1))
    taskParameters['forcedNoGo'] = float(values.get('-NoGo-', 0.0))  # Defaulting to 0 if missing

    taskParameters['x_FS1'] = float(values.get('-FS1X-', 0.45))
    taskParameters['y_FS1'] = float(values.get('-FS1Y-', 0.4))
    taskParameters['FS1Trials'] = float(values.get('-FS1Trials-', 0.1))

    taskParameters['x_HS1'] = float(values.get('-HS1X-', 0.0))
    taskParameters['y_HS1'] = float(values.get('-HS1Y-', -0.2))
    taskParameters['HS1Trials'] = float(values.get('-HS1Trials-', 0.1))

    taskParameters['x_PPC'] = float(values.get('-PPCX-', -0.15))
    taskParameters['y_PPC'] = float(values.get('-PPCY-', -1.3))
    taskParameters['PPCTrials'] = float(values.get('-PPCTrials-', 0.1))

    taskParameters['x_DC'] = float(values.get('-DCX-', 4.0))
    taskParameters['y_DC'] = float(values.get('-DCY-', -4.0))
    taskParameters['DCTrials'] = float(values.get('-DCTrials-', 0.7))  # Ensure it has a default value
    
    


    # Determine session mode from checkboxes.
    if values['-CW-'] and values['-CCW-']:
        sg.popup("Error: Select only one session type.")
        raise ValueError("Both session modes selected.")
    elif not (values['-CW-'] or values['-CCW-']):
        sg.popup("Error: No session mode selected.")
        raise ValueError("No session mode selected")
    else:
        if values['-CW-']:
            taskParameters['sessionMode'] = 'clockwise'
        else:
            taskParameters['sessionMode'] = 'counterclockwise'
    # Conditioning mode drop-down; default to Operant.
    taskParameters['conditioningMode'] = values['-MODE-'] if values['-MODE-'] != "" else "Operant"
    return taskParameters

##################### Main GUI #####################
def the_gui():
    sg.theme('Default1')
    textWidth = 23
    inputWidth = 15
    settings = load_settings(SETTINGS_FILE, DEFAULT_SETTINGS)
    # Initialize flag and dictionary for laser parameters.
    laserParamsSet = False
    laserParams = {}
    left_column = [
        [sg.Text('Number of Trials', size=(textWidth,1)), sg.Input(300, size=(inputWidth,1), key='-NumTrials-')],
        [sg.Text('Start Probes', size=(textWidth,1)), sg.Input(200, size=(inputWidth,1), key='-StartProbes-')],
        [sg.Text('End Probes', size=(textWidth,1)), sg.Input(220, size=(inputWidth,1), key='-EndProbes-')],
        [sg.Text('Trial Duration (s)', size=(textWidth,1)), sg.Input(default_text=6.9893, size=(inputWidth,1), key='-TrialDuration-')],
        [sg.Text('Lick free seconds', size=(textWidth,1)), sg.Input(2.9906, size=(inputWidth,1), key='-LickFree-')],
        [sg.Text('Go Probability', size=(textWidth,1)), sg.Input(default_text=0.5, size=(inputWidth,1), key='-GoProbability-')],
        [sg.Text('False Alarm Timeout (s)', size=(textWidth,1)), sg.Input(default_text=7.9974, size=(inputWidth,1), key='-FalseAlarmTimeout-')],
        [sg.Text('Stimulus Onset (s)', size=(textWidth,1)), sg.Input(default_text=1.00807428, size=(inputWidth,1), key='-StimTime-')],
        [sg.Text('Stim Duration (s)', size=(textWidth,1)), sg.Input(default_text=1.00807428, size=(inputWidth,1), key='-StimDuration-')],
        [sg.Text('Reward Window Onset (s)', size=(textWidth,1)), sg.Input(default_text=2.01614856, size=(inputWidth,1), key='-RewardWindowOnset-')],
        [sg.Text('Reward Window Duration (s)', size=(textWidth,1)), sg.Input(default_text=0.50403714, size=(inputWidth,1), key='-RewardWindowDuration-')],
        # sg.Check('Reward All Go Trials?', key='-RewardAllGos-', default=False)],
        [sg.Text('Sample Rate (Hz)', size=(textWidth,1)), sg.Input(default_text=20000, size=(inputWidth,1), key='-SampleRate-'),
         sg.Check('Downsample?', key='-DownSample-', default=False)],
        [sg.Text('Save Path', size=(textWidth,1)), sg.Input(os.path.normpath(r'C:\Users\labemanuel\Desktop\Ronghao_Zhang\data'), size=(20,1), key='-SavePath-'),
         sg.Check('Save?', key='-Save-', default=True)],
        [sg.Text('Animal ID', size=(textWidth,1)), sg.Input(size=(20,1), key='-Animal-')],
        [sg.Text("Conditioning Mode:", size=(textWidth,1)), sg.Combo(["Operant", "Classical"], default_value="Operant", key='-MODE-', size=(inputWidth,1))],
        [sg.Checkbox("Clockwise Session", key='-CW-', default=True),
         sg.Checkbox("Counter-Clockwise Session", key='-CCW-', default=False)],
        [sg.Checkbox("Sculpt Behavior (Forced No-go)?", key='-SculptBehavior-', default=True)],
        [sg.Text('Dispensed Volume (mL)', size=(textWidth,1)), sg.Input(default_text=0.0061, size=(inputWidth,1), key='-DispensedVolume-')],
        [sg.Text('25 ms = 0.0012, 40 ms = 0.0041, 50 ms = 0.004, 100 ms = 0.0126')],
        # New checkbox for using laser.
        [sg.Checkbox("Use Laser & XY", key='-Laser-', default=False)],
        [sg.Column([[sg.Button('Run Task', size=(20,2))]], justification='center')],
        [sg.Column([[sg.Button('Pause/Resume', key='-PAUSE-', size=(20,1))]], justification='center')],
        [sg.Column([[sg.Button('End Session', key='-END-', size=(20,1))]], justification='center')],
        [sg.Column([[sg.Button('Update Parameters'), sg.Button('Exit'), sg.Button('Setup DAQ')]], justification='center'),
         sg.Input(key='-LoadParams-', visible=False, enable_events=True),
         sg.FileBrowse('Load Parameters', target='-LoadParams-', initial_folder='D:\\')]
    ]
    right_column = [
        [sg.Output(size=(70,30), key='-OUTPUT-')]
    ]
    layout = [
        [sg.Column(left_column), sg.Column(right_column)]
    ]
    window = sg.Window('Brush Task', layout, finalize=True)
    
    di_task, ao_task, do_task, daqStatus = None, None, None, None

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'Update Parameters':
            try:
                taskParameters = updateParameters(values)
                print('Parameters updated')
            except ValueError:
                continue
        if event == 'Setup DAQ':
            s_window = create_settings_window(settings)
            s_ev, s_vals = s_window.read(close=True)
            if s_ev == 'Save':
                save_settings(SETTINGS_FILE, settings, s_vals)
        # Monitor the Laser checkbox before "Run Task" is pressed.
        if values['-Laser-'] and not laserParamsSet:
            laser_vals = createLaserSettingsWindow()
            if laser_vals is not None:
                laserParams = laser_vals
                laserParamsSet = True
                print("Laser settings updated.")
            else:
                window['-Laser-'].update(value=False)
                print("Laser settings canceled. Laser option disabled.")
        if event == 'Run Task':
            try:
                taskParameters = updateParameters(values)
                print('Parameters updated')
            except ValueError:
                continue
            # Merge previously obtained laser parameters if set.
            if laserParamsSet:
                taskParameters.update(laserParams)
                taskParameters['laserParamsSet'] = True
                print("Merged Laser Params:", taskParameters.get('x_Ref'), taskParameters.get('y_Ref'))
            try:
                if daqStatus != 'task':
                    if di_task is not None:
                        di_task.close()
                    if do_task is not None:
                        do_task.close()
                    di_task, ao_task, do_task, daqStatus = setupDaq(settings, taskParameters)
            except NameError:
                di_task, ao_task, do_task, daqStatus = setupDaq(settings, taskParameters)
            if taskParameters['conditioningMode'] == "Operant":
                threading.Thread(target=runTask, args=(di_task, ao_task, do_task, taskParameters, settings), daemon=True).start()
            else:
                threading.Thread(target=runTask, args=(di_task, None, do_task, taskParameters, settings), daemon=True).start()
        if event == '-PAUSE-':
            if pause_event.is_set():
                pause_event.clear()
                window['-PAUSE-'].update("Resume")
                print("Task paused.")
            else:
                pause_event.set()
                window['-PAUSE-'].update("Pause/Resume")
                print("Task resumed.")
        if event == '-END-':
            end_session_event.set()
            print("End Session button pressed. Session will end after the current trial.")
        if event == '-LoadParams-':
            param_file = values['-LoadParams-']
            if os.path.isfile(param_file):
                try:
                    with open(param_file, 'rb') as f:
                        in_data = pickle.load(f)
                    tempParameters = in_data['taskParameters']
                    print(f'Updating parameters from {param_file}')
                    window.Element('-NumTrials-').Update(value=tempParameters['numTrials'])
                    window.Element('-SampleRate-').Update(value=tempParameters['Fs'])
                    window.Element('-DownSample-').Update(value=tempParameters['downSample'])
                    window.Element('-TrialDuration-').Update(value=tempParameters['trialDuration'])
                    window.Element('-FalseAlarmTimeout-').Update(value=tempParameters['falseAlarmTimeout'])
                    window.Element('-GoProbability-').Update(value=tempParameters['goProbability'])
                    window.Element('-StimTime-').Update(value=tempParameters['stimTime'])
                    window.Element('-StimDuration-').Update(value=tempParameters['stimDuration'])
                    window.Element('-RewardWindowOnset-').Update(value=tempParameters['rewardWindowOnset'])
                    window.Element('-RewardWindowDuration-').Update(value=tempParameters['rewardWindowDuration'])
                    window.Element('-SavePath-').Update(value=tempParameters['savePath'])
                    window.Element('-Save-').Update(value=tempParameters['save'])
                    window.Element('-Animal-').Update(value=tempParameters['animal'])
                    window.Element('-LickFree-').Update(value=tempParameters['lickFree'])
                    if tempParameters.get('sessionMode','clockwise') == 'clockwise':
                        window.Element('-CW-').Update(value=True)
                        window.Element('-CCW-').Update(value=False)
                    else:
                        window.Element('-CW-').Update(value=False)
                        window.Element('-CCW-').Update(value=True)
                except Exception as e:
                    print(f'Error loading parameters from file: {e}')
            else:
                print('Invalid file selected for loading parameters.')
    window.close()

if __name__ == '__main__':
    print('starting up')
    the_gui()
    print('Exiting Program')
