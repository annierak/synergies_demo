import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animate

a = np.random.randn(5,100)

dt = 0.1
simulation_time = 10

counter = 0
file_name = 'example_video'
frame_rate = 20
times_real_time = 1 # seconds of real time / sec of video
capture_interval = int(np.ceil(times_real_time*(1./frame_rate)/dt))
t=0

plt.ion()
fig=plt.figure()

FFMpegWriter = animate.writers['ffmpeg']
metadata = {'title':file_name,}
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, file_name+'.mp4', 500)

# plt.show()
lines = plt.plot(a.T)

while t<simulation_time:
    a[:,:-1] = a[:,1:]
    a[:,-1] = np.random.randn(5)
    for row,line in enumerate(lines):
        # print(a[row,:])
        line.set_ydata(a[row,:])
    plt.draw()
    plt.pause(0.01)
    writer.grab_frame()
    t+=dt

writer.finish()
