import os
import subprocess
import pdb
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# Custom imports from local modules
from ..utils.utils import save_list_to_txt, load_list_from_txt, write_pandas_csv, read_csv_pandas
from ..transforms import EventReduceVoxel, EventAccumulation

# Constants
MAGIC_NUMBER = "2211feca"
HEADER = (
    "% camera_integrator_name Prophesee\n"
    "% date 2023-07-18 14:28:24\n"
    "% evt 2.1\n"
    "% format EVT21;height=320;width=320\x00% generation 4.2\n"
    "% geometry 320x320\x00% integrator_name Prophesee\n"
    "% plugin_integrator_name Prophesee\n"
    "% plugin_hame hal_plugin_prophesee\n"
    "% sensor_generaation 4.2\n"
    "% serial_number 00050540\n"
    "% system_ID 49\n"
    "% end\n"
)
COLUMNS = 320
ROWS = 320


class DVS:
    def __init__(self, path=None, imu_ts=None, verbose=False, config={}, overwrite=True, vis_mode=False):
        # Initialized Variables
        self.config = config 
        self.data = [] 

        # Initialize Transform
        self.evs_reduce_fn = EventReduceVoxel()
        self.evs_acc_fn = EventAccumulation(
            num_bins=self.config.get("event_windows", 1),
            polarities_mode=self.config.get("event_polarities_mode", "substract"),
            polarities=self.config.get("event_polarities", 1),
            decay_constant=self.config.get("event_decay_constant", 1.0),
            mode=self.config.get("event_accumulation", "addition")
        )
        
        if vis_mode: return 
        
        # Initialize paths
        assert os.path.exists(path), f"FileNotFound Error {path}"
        self.path = Path(path)
        self.base_dir = self.path.parent
        self.basename = self.path.stem 
        self.dvs_tts_path = self.base_dir / f"{self.basename}_dvs_tts.txt"
        self.evs_mod_path = self.base_dir / f"{self.basename}_mod.raw"
        self.evs_syn_path = self.base_dir / f"{self.basename}_syc.csv"

            
        # Synchronize and Load Data
        files_to_check = [self.dvs_tts_path, self.evs_syn_path, self.evs_mod_path]
        for file_path in files_to_check:
            if (os.path.exists(file_path) and overwrite) or not os.path.exists(file_path):     
                events, dvs_tts = dvs_index_rawfile(path, self.evs_mod_path, self.dvs_tts_path) 
                if imu_ts is not None : dvs_synchronize_timestamps(imu_ts, dvs_tts, events, self.evs_syn_path) 
                break
        
        # Load synchronized data into attributes
        self.events = read_csv_pandas(self.evs_syn_path, dtype={
            'x': 'int32',
            'y': 'int32',
            'p': 'int8',
            't': 'float32'
        })

        self.use_slice_with_dt = config["frequency"] == "dvs"
        self.event_dt_ms = config["event_dt_ms"]  
        
        if verbose: print(f"Indexed DVS : {path}")

    def __len__(self):
        if self.use_slice_with_dt:
            time_window = (self.events["t"].iloc[-1] - self.events["t"].iloc[0])*1000
            number_of_windows = round(time_window/self.event_dt_ms) 
            return number_of_windows 
        try:
            return len(self.events)
        except:
            return len(self.data)

    def add_data(self, data):    
        b, wc, h, w = data.shape 
        data_reshaped = data.reshape(b, self.config["event_windows"], self.config["event_polarities"], h, w).numpy()  
        data_reshaped /= 2
        data_reshaped += 0.5 
        data_reshaped *= 255.0
        data_transposed = np.transpose(data_reshaped.sum(1), (0, 2, 3, 1)) 
        self.data.append(self.evs_reduce_fn(data_transposed))  

    def reset(self):    
        del self.data
        self.data = [] 

    def get_empty_array(self):
        return np.zeros((self.config["event_windows"], self.config["event_polarities"], ROWS, COLUMNS), dtype=np.uint8)

    def slice_with_dt(self, index):
        base_value = self.events["t"].iloc[0]

        t_end = base_value + index*self.event_dt_ms/1000
        t_start = t_end - self.event_dt_ms/1000

        return self.get_closest(t_start, t_end)

    def get_closest(self, t_start, t_end):   
        idx = np.where(((self.events["t"] - t_start)  > 0) & ((self.events["t"] - t_end) <= 0))[0]
        if len(idx) == 0:  
            fixed_ts = np.linspace(t_start, t_end, self.config["event_windows"])
            return {"t_start": t_start, "t_end": t_end, "fixed_ts": fixed_ts, "data": self.get_empty_array()}
        else:
            return self.__getitem__(idx) 

    def __getitem__(self, index):   
        data_dict = self.evs_acc_fn(self.get_empty_array(),  self.events.iloc[index])  
        return {"t_start": data_dict["t_start"], "t_end": data_dict["t_end"], "fixed_ts": data_dict["timestamps"], "data" : data_dict["data"]}

    def cut_recording(self, index_array):   
        tmp_events = self.events.loc[index_array].reset_index(drop=True) 
        del self.events
        self.events = tmp_events

    def plot(self, ax, item, reduce=False, transpose=False, save_path=None): 
        data = item["data"] 

        if transpose:
            data = np.transpose(data, (0, 2, 3, 1))[0]

        if reduce:
            data = self.evs_reduce_fn(data)[0]
            data = data.clip(0, 1)

        if self.config["event_polarities"] > 1: 
            composite_image = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
            composite_image[..., 0] = data[..., 0]  # Red 
            composite_image[..., 1] = data[..., 1]  # Blue 
            data = composite_image 

        data = data.clip(0, 1)

        # Plot    
        ax.clear()  
        ax.imshow((1 - data) * 255, vmin=0, vmax=255, cmap=cm.gray, label='DVS')
        ax.axis('off') 
        # ax.set_title(item['t_end'])

        # Add a border around the image
        height, width = data.shape[:2] 
        border = Rectangle((0, 0), width, height, linewidth=6, edgecolor='grey', facecolor='none')
        ax.add_patch(border)

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            print(f"Plot saved to {save_path}")

# DVS .RAW Reader
# def dvs_copy_file_fromgdrive(path): 
#     filename = os.path.basename(path)
#     command = f"cp ~/googledrive/2025_SONY_RAP/vicon_dummy_dataset6/DVS/{filename} /datasets/pbonazzi/sony-rap/glc_dataset/vicon_aggregate/data/dvs/"
#     subprocess.run(command, shell=True, check=True) 

def dvs_index_rawfile(path, evs_mod_path, dvs_tts_path):   
    dvs_tts = [] 
    events = pd.DataFrame() 
    raw_file = open(path, mode="rb")
    raw_file.seek(0,2)
    num_bytes = raw_file.tell()  
    raw_file.seek(0,0)

    out_file_evs_mod = open(evs_mod_path, 'wb')
    out_file_evs_mod.write(HEADER.encode('ascii'))
    for i in tqdm(range(0, num_bytes>>3-1)):
        dataLSB =  raw_file.read(4)  
        #  Timestamps
        if (dataLSB.hex() == MAGIC_NUMBER):
            ts_str =  raw_file.read(4)
            ts_lsb = int.from_bytes(ts_str, byteorder='little')
            ts_str =  raw_file.read(4)  
            ts_msb = int.from_bytes(ts_str, byteorder='little') 
            ts = round(((ts_msb<<32) + ts_lsb) * 10e-9, 8) 
        else:
            # Decode the 8 bytes into four components
            dataMSB =  raw_file.read(4)   
            out_file_evs_mod.write(dataMSB)
            out_file_evs_mod.write(dataLSB)# TODO: Ask why is it reversed 

    raw_file.close()
    out_file_evs_mod.close() 

    # Events
    from metavision_core.event_io import EventsIterator 
    mv_iterator = EventsIterator(input_path= str(evs_mod_path)) 
    for ev in mv_iterator:   
        events = pd.concat([events, pd.DataFrame(ev)]) 

    # Trigger Timestamps
    external_triggers = mv_iterator.get_ext_trigger_events() 
    for trig in external_triggers:
        if trig[0] == 1:
            dvs_tts.append(trig[1])
    save_list_to_txt(dvs_tts, dvs_tts_path)   
    
    return events, dvs_tts

def dvs_synchronize_timestamps(imu_ts, dvs_tts, events, evs_syn_path):  
    synch_raw = pd.DataFrame()
    length = len(dvs_tts) 
    
    for index in tqdm(range(0, length)):

        t0 = dvs_tts[index-1] if index > 0 else 0
        t1 = dvs_tts[index]
        
        raw = events[(events["t"] > float(t0)) & (events["t"] < t1)].copy()
        raw["t"] = raw["t"].astype(float)  # Ensure the column is float
        
        imu_time = imu_ts[index].astype(float)  # Ensure imu_ts values are float
        t1_float = float(t1)  # Ensure t1 is float
        
        raw["t"] = ((raw["t"] - t1_float) * 1e-6) + imu_time 
        synch_raw = pd.concat([synch_raw, raw])
    
    idx_last_sy_ev = len(synch_raw)
    if len(dvs_tts) == 0:
        t1_float = 0
        imu_time = imu_ts[0]
        
    if len(synch_raw) != len(events) :
        raw = events.iloc[idx_last_sy_ev:].copy()
        raw["t"] = raw["t"].astype(float) 
        raw["t"] = ((raw["t"] - t1_float) * 1e-6) + imu_time 
        synch_raw = pd.concat([synch_raw, raw])
        
    write_pandas_csv(synch_raw, evs_syn_path) 
    
if __name__ == '__main__':
    base_path = "/datasets/pbonazzi/sony-rap/glc_dataset/vicon_aggregate/data/"
    from .imu import IMU
    
    rec_id = 1
    imu = IMU(path=os.path.join(base_path, "imu", f"IMU{rec_id}.RAW"), verbose=False, overwrite=False) 
    dvs = DVS(path=os.path.join(base_path, "dvs", f"DVS{rec_id}.RAW"), imu_ts=imu.timestamps,
              config={"event_windows": 1, 
                      "event_dt_ms": 20,
                      "event_polarities": 1, 
                      "event_polarities_mode": "substract",
                      "event_accumulation": "addition", 
                      "event_decay_constant": 1.0,
                      "frequency": "dvs"}, 
              verbose=False, overwrite=False) 
    # from ..utils import is_pickleable
    # is_pickleable(rec)

    pdb.set_trace()