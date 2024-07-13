import numpy as np
import scipy.fft as fft

def video_freq_projection(frames, fps, T=None, m=None): 
    """
    frames : ndarray
        3D numpy array with time dimension corresponding to axis 0. 
    fps : float
        frames per second
    T : float
        Time period (in seconds) of the desired frequency.
    """
    N  = len(frames)
    ts = range(len(frames))

    if T is None and m is None:
        projected_video = np.mean(frames, axis=0)

    elif m is None: 
        m = T*fps
    
    An = np.sum([frames[t]*np.exp(-1j*2*np.pi*t/m) for t in ts], axis=0) / N
    projected_video = np.real(np.array([An*np.exp(1j*2*np.pi*t/m) for t in ts]))

    return projected_video

def video_fft(frames, fps=1, apply_fftshift=False):
    frames_fft = fft.rfft(frames, axis=0)
    freqs = fft.rfftfreq(len(frames), 1/fps)
    return freqs, frames_fft

def video_ifft(freqs, frames_fft, fps=1, freq_loc=None, sigmas=1):
    if freq_loc is not None:
        delta = (freqs[1] - freqs[0])*sigmas
        mask = np.zeros_like(frames_fft)
        mask[np.abs(np.abs(freqs) - freq_loc) < delta,:,:] = 1
        frames_fft = mask*frames_fft

    video_ifft = fft.irfft(frames_fft, axis=0)
    ts = np.arange(len(frames_fft))/fps
    return ts, video_ifft

if __name__=="__main__":
    import os
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.signal import find_peaks

    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    height_fields = np.load(dir_path.joinpath(f"Imagenes{os.sep}0611-toroide_forzado (height fields frames 1150-1350).npy"))

    vid_length, vid_width, vid_height =  height_fields.shape
    R = vid_width // 2

    # Con video_fft
    freqs, vidfft = video_fft(height_fields, fps=1000, apply_fftshift=True)

    peaks,_ = find_peaks(np.abs(vidfft[:,R,R]), height=0.02)
    f = freqs[peaks][0]  # guarda frecuencia pico

    ts, vidifft = video_ifft(freqs, vidfft, fps=1000, freq_loc=f) # Transformada inversa quedandose con componentes en f

    plt.title("Con video_ifft")
    plt.imshow(vidifft[:,:,R])
    plt.show()
    
    # Con video_freq_projection
    projected_video = video_freq_projection(height_fields, 1000, T = 1/f)
    
    plt.title("Con video_freq_projection")
    plt.imshow(projected_video[:,:,R])
    plt.show()

    # Gráfico interactivo
    plt.ion()

    fig = plt.figure()
    ax = plt.subplot()

    l, = ax.plot(vidifft[0,:,R]) 
    ax.set_ylim(np.min(vidifft),np.max(vidifft))

    for frame in vidifft:
        l.set_ydata(frame[:,R])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(4e-3)

    plt.ioff()


