import matplotlib.pyplot as plt
from pydata.video import Video
from pyfcd.layer import Layer
from pydata.image import *
from pyfcd.fcd import FCD
from scipy.fft import fftfreq, fft, rfft, irfft, rfftfreq
from scipy.signal import find_peaks

if __name__ == '__main__':
    template = Image("E:/Ignacio Hernando/29_10/20241029-template-grande_202410_0956/20241029-template-grande_202410_0956000001.bmp", roi=(215, 342, 454, 454))  
    template.center_toroid()
    template.rotated(angle=45)

    reference = Image("E:/Ignacio Hernando/29_10/20241029-referencia_202410_0952/20241029-referencia_202410_0952000001.bmp", roi=template.roi)
    layers = [Layer(3.334e-2, "Air"), Layer(1.2e-2, "Acrylic"), Layer(4.34e-2, "Distilled_water"), Layer(80e-2, "Air")]
    fcd = FCD(reference.windowed(), square_size=0.0022, layers=layers)
    mask = reference.make_circular_mask(radius=190)

    video = Video("E:/Ignacio Hernando/29_10/impulso_1_grande_202410_1010", start_frame=1600)

    frames = []
    for frame in video.play(end_frame=3000)
        frame.track(template.processed)
        reference.roi = frame.roi  
        frame.masked(mask, reference.windowed())
        height_field = fcd.analyze(frame.windowed(), full_output=False)
        frames.append(height_field)
        print(video.current_frame_index)
    
    ### Análisis de los datos procesados. ###
    def video_fft(frames, fps=1, apply_fftshift=False):
        frames_fft = rfft(frames, axis=0)
        freqs = rfftfreq(len(frames), 1/fps)
        return freqs, frames_fft

    def amplitude(freqs, frames_fft, fps=1, freq_loc=None, sigmas=1):
        differences = np.abs(np.abs(freqs) - freq_loc)
        index = np.argwhere(differences == min(differences))[0][0]
        print(f"Diferencia con la frecuencia deseada es de: {differences[index]:.2e} Hz. ")
        return np.abs(frames_fft[index])

    frames = np.array(frames)
    R = video.current_frame.roi[2]//2
    f = 3.38 # 5.37

    freqs, vidfft = video_fft(frames, fps=500, apply_fftshift=True)
    amplitudes = amplitude(freqs, vidfft, fps=500, freq_loc=f) 
    np.save(f"amplitude_{f}Hz_{video.frames[0][:-3]}", amplitudes)
    plt.imshow(amplitudes)
    plt.show()
