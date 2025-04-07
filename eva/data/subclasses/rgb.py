import os
import pdb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
from ..utils.utils import save_list_to_txt, load_list_from_txt
from ..transforms import CorrectWhiteBalance 

COLUMNS = 640
ROWS = 480

class RGB:
    def __init__(self, path=None, overwrite=True, verbose=False, vis_mode=False, rgb_windows=1):
        # Initialized Variables 
        self.timestamps = []
        self.data = [] 
        self.rgb_windows = rgb_windows

        if vis_mode:
            return 
        
        assert os.path.exists(path), f"FileNotFound Error {path}"
        self.path = path
        self.timestamps_path = path.replace(".RAW", "_time.txt") 
        self.path_to_img_dir = path.replace(".RAW", "") 

        if overwrite and os.path.exists(self.timestamps_path): os.remove(self.timestamps_path) 
        if overwrite and os.path.exists(self.path_to_img_dir): shutil.rmtree(self.path_to_img_dir)
        os.makedirs(self.path_to_img_dir, exist_ok=True)

        # Initialize File 
        if not os.path.exists(self.timestamps_path): rgb_indexfile(path, self.timestamps_path, self.path_to_img_dir) 
        self.timestamps = np.asarray(load_list_from_txt(self.timestamps_path)).astype(float) 
        self.indexes_for_images = np.arange(0, len(self.timestamps))
        if verbose:  print(f"Indexed RGB : {path}") 
        self.correctwhitebalance = CorrectWhiteBalance() 

    def __len__(self):
        return len(self.indexes_for_images) - 1
        
    def __repr__(self):
        description = "RGB(samples="+str(self.__len__())+")"
        return description

    def cut_recording(self, index_array):  
        self.timestamps = self.timestamps[index_array] 
        self.indexes_for_images = self.indexes_for_images[index_array] 

    def get_closest(self, t_start, t_end):    
        idx = np.where((self.timestamps - t_end <= 0)&(self.timestamps - t_start  >= 0))[0]
        if len(idx) == 0:
            print(0)
            return {"t": 0.0, "data" : np.transpose(np.zeros([self.rgb_windows, COLUMNS, ROWS, 3]), (0, 3, 1, 2))}
        else:
            return self.__getitem__(np.max(idx))

    def get_next_async(self, t_start, t_end):    
        idx = np.where((self.timestamps - t_end <= 0)&(self.timestamps - t_start  >= 0))[0]
        if len(idx) == 0: 
            return {"t": 0.0, "data" : np.transpose(np.zeros([self.rgb_windows, COLUMNS, ROWS, 3]), (0, 3, 1, 2))}
        else:
            return self.__getitem__(np.max(idx))

    def __getitem__(self, index): 

        # Timestamp 
        t_end = self.timestamps[index] 
        t_start = t_end if index==0 else self.timestamps[index-1] 

        # RGB load images
        img_list = []
        for i in range(0, self.rgb_windows): 
            file_path = os.path.join(self.path_to_img_dir, f"{(self.indexes_for_images[index-i]):04}.png") 
            if index == 0 and i > 0:
                img = np.zeros_like(img)
            else:
                img = cv2.imread(file_path)
                img = self.correctwhitebalance(img)
                cv2.destroyAllWindows()
            img_bgr = img[..., ::-1]  # Reverse the color channels
            img_transposed = np.transpose(img_bgr, (2, 0, 1))
            img_list.append(img_transposed)

        img_list = np.stack(img_list)
        return {"t_start": t_start, "t_end": t_end, "data" : img_list}

    def add_data(self, data):    
        self.data.append(data.numpy())  

    def reset(self):    
        del self.data
        self.data = [] 

    def plot(self, ax, item, transpose=True, resize=False, save_path=None):
        ax.clear()  
        data = item["data"]
        if transpose:    
            data = np.transpose(data[0], (1, 2, 0))
        if resize: 
            data = cv2.resize(data, (320, 320))

        ax.imshow(data, label='RGB Data')
        ax.axis('off')  
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1) 

# RGB IndexFile
def rgb_indexfile(path, timestamps_path, path_to_img_dir): 

    raw_file = open(path, mode="rb")
    raw_file.seek(0,2)
    num_bytes = raw_file.tell()  
    if num_bytes == 0:
        print("File has 0 bytes")
        pdb.set_trace() 
    raw_file.seek(0,0)

    timestamps = [] 
    
    for i in range(0, num_bytes//(2*ROWS*COLUMNS)): 
        # timestamps
        ts_str = raw_file.read(4)
        ts_lsb = int.from_bytes(ts_str, byteorder='little')
        ts_str = raw_file.read(4)  
        ts_msb = int.from_bytes(ts_str, byteorder='little')
        ts = round(float(((ts_msb<<32) + ts_lsb) * 10e-9), 8)
        timestamps.append(ts)    

        image = np.zeros([ROWS, COLUMNS], dtype='u1')
        x, y = 0, 0
        for _ in range(ROWS*COLUMNS):
            pixel_str = raw_file.read(2)
            pixel_bin = int.from_bytes(pixel_str, byteorder='little')
            image[y, x] = (pixel_bin >> 2)
            x += 1
            if ((x % COLUMNS) == 0):
                x = 0
                y += 1
                if ((y % ROWS) == 0):
                    break

        img = cv2.cvtColor(image, cv2.COLOR_BayerRGGB2BGR).copy() 
        file_path = os.path.join(path_to_img_dir, f"{i:04}.png") 
        cv2.imwrite(file_path, img)  
    
    raw_file.close()
    save_list_to_txt(timestamps, timestamps_path)    

if __name__ == '__main__':
    base_path = "/datasets/pbonazzi/sony-rap/glc_dataset/2_vicon_dummy_dataset2/"
    rec = RGB(path=os.path.join(base_path, f"RGB2.RAW"), verbose=False, overwrite=False) 

    pdb.set_trace()