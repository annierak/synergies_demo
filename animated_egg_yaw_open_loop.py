import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #This needs to be placed before importing any sub-packages
import matplotlib.animation as animate
import figurefirst as fifi
import flylib as flb
import cairo
import rsvg
import networkx as nx
import plotting_utls as pltutls
# import faulthandler; faulthandler.enable()

fly_num = 1548
disp_window_size = 4
corr_window_size = 1
frames_per_step=1

def render_svg_to_png(svg_data,filename):
    # Render
    svg = rsvg.Handle(data=svg_data)
    img = cairo.ImageSurface(cairo.FORMAT_ARGB32,
    svg.props.width,
    svg.props.height)
    ctx = cairo.Context(img)
    svg.render_cairo(ctx)
    img.write_to_png(filename)



try:
    fly = flb.NetFly(fly_num,rootpath='/home/annie/imager/media/imager/FlyDataD/FlyDB/')
except(OSError):
    fly = flb.NetFly(fly_num)

flydf = fly.construct_dataframe()



dt = frames_per_step*(flydf['t'][1]-flydf['t'][0])
corr_window_size = np.floor(corr_window_size/dt)*dt
disp_window_size = np.floor(disp_window_size/dt)*dt

steps_per_disp_window = int(disp_window_size/dt)
steps_per_corr_window = int(corr_window_size/dt)


filtered_muscle_cols = \
['iii1_left', 'iii3_left',
 'i1_left',  'i2_left',
 'hg1_left', 'hg2_left', 'hg3_left', 'hg4_left',
 'b1_left', 'b2_left', 'b3_left',
 'iii1_right', 'iii3_right',
 'i1_right', 'i2_right',
 'hg1_right', 'hg2_right', 'hg3_right', 'hg4_right',
 'b1_right', 'b2_right', 'b3_right' ]


general_sorted_keys = sorted(fly.ca_cam_left_model_fits.keys())
print(sorted(fly.ca_cam_left_model_fits.keys()))

sorted_keys = []

for key in general_sorted_keys:
    key2= key+'_right'
    key3= key+'_left'
    sorted_keys.append(key2)
    sorted_keys.append(key3)

cull_list = [('left', 'bkg'),('right', 'bkg'),
            ('left', 'iii24'),('right', 'iii24'),
            ('left', 'nm'),('right', 'nm'),
            ('left', 'pr'),('right', 'pr'),
            ('left', 'tpd'),('right', 'tpd')]

for cull in cull_list:
    sorted_keys.remove(cull[1]+'_'+cull[0])
#[sorted_keys.remove(cull) for cull in cull_list]



#print(sorted_keys)
#raw_input(' ')

dpi = 600


#--------------HERE IS WHERE YOU SPECIFY THE TRIAL-----------------------
trial_str = 'cl_blocks, g_x=-1, g_y=0, b_x=0, b_y=0, ch=1'
counter=np.where(flydf['stimulus']==trial_str)[0][0]
t=flydf.iloc[counter]['t']
#------------------------------------------------------------------------

plt.ion()
fig = plt.figure(1,figsize=(6,12))#,dpi=dpi)
filename = 'f_egg_entire_fly_'+str(fly_num)+trial_str
layout = fifi.FigureLayout('graph_layout.svg')#,make_mplfigures=True)
metadata = {'title' : filename,}
# im = plt.imshow(np.random.randn(10,10))


# time_window_inds = (flydf['t']>t)&(flydf['t']<=t+corr_window_size)
print(counter+steps_per_corr_window)
time_window_inds = np.arange(counter,counter+steps_per_corr_window)

kin_values_by_t = np.zeros(steps_per_disp_window)

nonan_amp_diff = flydf['amp_diff'][~np.isnan(flydf['amp_diff'])]
min_kin,max_kin = np.percentile(nonan_amp_diff,0.5),np.percentile(nonan_amp_diff,99.5)

# print(min_kin,max_kin)
# raw_input(' ')

kinax = plt.subplot2grid((8,2),(1,0),colspan=2,rowspan=2)
lines, = plt.plot(kin_values_by_t)
plt.title('Wing Kinematics')
plt.xlabel('Time (s)')
plt.ylabel('Wing Amplitude Difference')
plt.ylim([min_kin,max_kin])
plt.tick_params(axis='x', which='major', labelsize=8)
# plt.tick_params(axis='both', which='minor', labelsize=8)

plt.text(0.5,.9,'Trial Type:',fontsize=14,transform=plt.gcf().transFigure,
    horizontalalignment='center')
trial = plt.text(0.5,0.88,'trial',transform=plt.gcf().transFigure,
    horizontalalignment='center')


frame_rate = 5
video_time = flydf.iloc[-1]['t']

ax = plt.subplot2grid((8,2),(3,0),colspan=2,rowspan=5)
im = plt.imshow(np.random.randn(80,60))
pltutls.strip_bare(ax)
plt.xlabel('Muscle Correlations')
plt.subplots_adjust(bottom=0.1, right=0.8, left= 0.2, top=.9)
#counter=0
FFMpegWriter = animate.writers['ffmpeg']
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, filename+'.mp4', dpi)

muscle_labels = np.zeros_like(filtered_muscle_cols)
for i,muscle in enumerate(filtered_muscle_cols):
    n1,n2 = muscle.split('_')
    muscle_labels[i] = '%s_%s'%(n2[0].capitalize(), n1)

# def updatefig(*args):
while flydf.iloc[counter]['stimulus']==trial_str:
    # global counter,t
    print(t)
    # time_window_inds = (flydf['t']>t)&(flydf['t']<=t+corr_window_size)
    time_window_inds = np.arange(counter,counter+steps_per_corr_window)

    #(1) --------Correlation Matrix----------
    state_mtrx = np.vstack([flydf[key][time_window_inds] for key in sorted_keys])
    #Watch out for muscles that have no activity
    off_muscle_inds = (np.sum(state_mtrx,axis=1)==0.)
    # Set them to a very small amt of activity so nan's are not created
    state_mtrx[off_muscle_inds] = 1e-8
    centered_mtrx = state_mtrx - np.mean(state_mtrx,axis = 1)[:,None]
    std_mtrx = centered_mtrx/np.std(centered_mtrx,axis = 1)[:,None]
    cor_mtrx = np.dot(std_mtrx,std_mtrx.T)

    G = nx.Graph()
    for i,lbl1 in enumerate(muscle_labels):
        for j,lbl2 in enumerate(muscle_labels):
            G.add_edge(lbl1,lbl2,weight = cor_mtrx[i,j])

    c_ex = layout.pathspecs['excitatory'].mplkwargs()['edgecolor']
    c_in = layout.pathspecs['inhibitory'].mplkwargs()['edgecolor']
    colors = [{True:c_ex,False:c_in}[G[e[0]][e[1]]['weight']>0.] for e in G.edges()]

    h = float(layout.layout_uh)
    pos_dict = {}
    for n in G.nodes():
        cx = float(layout.pathspecs[n]['cx'])
        cy = h-float(layout.pathspecs[n]['cy'])
        try:
            if 'transform' in layout.pathspecs[n].keys():
                # print(n_s)
                t1 = fifi.svg_to_axes.parse_transform(layout.pathspecs[n]['transform'])
                p = np.dot(t1,np.array([cx,cy,1]))
                pos_dict[n] = (p[0],p[1])
            else:
                pos_dict[n]  = (cx,cy)
        except KeyError:
            print n

    edges= G.edges()
    weights = [np.abs(G[e[0]][e[1]]['weight'])**2.5/(2e2) for e in edges]
    fig2 = plt.figure(2,figsize=(5,8))
    weights = np.array(weights)
    weights[np.isnan(weights)]=0.
    weights[np.isinf(weights)]=0.
    weights = list(np.array(weights).astype(str))
    plt.clf()
    nx.draw(G, pos = pos_dict,
            font_color = 'r', with_labels= True, width = weights,
            edge_color = colors, node_color = 'k', alpha = 0.3)
    fig2.savefig(filename+'.png')

    imported_image = plt.imread(filename+'.png',format='png')
    height,width,_ = np.shape(imported_image)
    imported_image = imported_image[int(np.floor(height/10)):int(np.floor(height-height/8)),:,:]
    im.set_array(imported_image)

    x_values = np.linspace(t,t+disp_window_size,steps_per_disp_window)


    #(2) Wing Kinematics
    kin_value = flydf.iloc[counter]['amp_diff']
    if counter<steps_per_disp_window:
        kin_values_by_t[counter] = kin_value
    else:
        try:
            hl.remove()
        except(NameError):
            pass
        kin_values_by_t[:-1] = kin_values_by_t[1:]
        kin_values_by_t[-1] = kin_value
        kinax.set_xlim((t,t+disp_window_size))
        lines.set_ydata(kin_values_by_t)
        lines.set_xdata(x_values)
        hl = kinax.axvspan(x_values[steps_per_disp_window-steps_per_corr_window-1], x_values[steps_per_disp_window-1],
            facecolor='b', alpha=0.5,transform=kinax.transAxes)
        writer.grab_frame()
        trial.set_text(flydf.iloc[counter]['stimulus'])
        kinax.set_aspect(2.5)


    counter+=1
    plt.draw()

    #Comment these out to save video
    # plt.pause(0.01)
    # plt.show()

    t+=dt
writer.finish()
# ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
# plt.show()
