import matplotlib.pyplot as plt
from pydata.video import Video
from pyfcd.layer import Layer
from pydata.image import *
from pyfcd.fcd import FCD
from scipy.fft import fftfreq, fft, rfft, irfft, rfftfreq
import copy
import h5py

if __name__ == '__main__':
    video = Video(roi=-1, start_frame=0)
    video.current_frame.center_toroid(use_sliders=True)   
    template = copy.deepcopy(video.current_frame)

    radius = int(input("Analyze Radius: "))   
    c = video.current_frame.roi[2] // 2
    R = int(radius / 1.41)  

    reference = Image(roi=video.current_frame.roi)
    layers = [Layer(3.422e-2, "Air"), Layer(1.2e-2, "Acrylic"), Layer(4.8e-2, "Distilled_water"), Layer(80e-2, "Air")]
    fcd = FCD(reference.processed, square_size=0.0022, layers=layers)  
    mask = reference.make_circular_mask(radius=radius)


    with h5py.File(f"{video.directory_path.split('/')[-1]}.h5", 'w') as f:
        dset = f.create_dataset('data', shape=(3072, template.roi[2], template.roi[2]), maxshape=(3072, 1024, 1024), dtype='float32')
        for frame in video.play():  # end_frame=15 
            frame.track(template.processed)
            reference.roi = frame.roi  
            frame.masked(mask, reference.processed)
            height_field = fcd.analyze(frame.processed, full_output=False)
            dset[video.current_frame_index-1] = height_field
            print(video.current_frame_index)
    
