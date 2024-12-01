import matplotlib.pyplot as plt
from pydata.video import Video
from pyfcd.layer import Layer
from pydata.image import *
from pyfcd.fcd import FCD
from scipy.fft import fftfreq, fft, rfft, irfft, rfftfreq
import copy
import h5py

if __name__ == '__main__':
    video = Video(roi=-1, start_frame=700)
    video.current_frame.center_toroid(use_sliders=True)   
    template = copy.deepcopy(video.current_frame)

    xs = []
    ys = []
    for frame in video.play(end_frame=1708):   
        frame.track(template.processed)
        xs.append(frame.roi[0] + frame.roi[2] //2 )
        ys.append(frame.roi[1] + frame.roi[2] //2 )
        if video.current_frame_index % 100 == 0:
            print(video.current_frame_index)

    times = np.arange(len(xs)) / video.fps * 1000

    # Subplot 1: Altura vs Frames
    plt.subplot(211)
    plt.plot(times, xs - np.mean(xs))
    plt.plot(times, ys - np.mean(ys))
    plt.xlabel("Tiempo [ms]")
    plt.ylabel("Altura [px]")

    # Subplot 2: FFT
    plt.subplot(212)
    freqs = fftfreq(len(times), times[1] - times[0]) * 1000
    ffts_x = fft(xs - np.mean(xs))
    ffts_y = fft(ys - np.mean(ys))
    plt.plot(freqs[:len(times)//2][::-1], np.abs(ffts_x)[:len(times)//2][::-1])
    plt.plot(freqs[:len(times)//2][::-1], np.abs(ffts_y)[:len(times)//2][::-1])
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [px]")

    # plt.ylim(0, 4000/2/2/1.5)
    
    plt.show()
