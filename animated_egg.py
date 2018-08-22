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
# import faulthandler; faulthandler.enable()

fly_num = 1548
corr_window_size = 5
frames_per_step=5
t=0

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

plt.ion()
fig = plt.figure(1,figsize=(5,8))
filename = 'f_egg'
layout = fifi.FigureLayout('graph_layout.svg')#,make_mplfigures=True)
metadata = {'title' : filename,}
# im = plt.imshow(np.random.randn(10,10))
counter=0


frame_rate = 20
counter =0
simulation_time = 4*60

#im = plt.imshow(np.random.randn(10,10))
#counter=0
FFMpegWriter = animate.writers['ffmpeg']
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, filename+'.mp4', 500)

muscle_labels = np.zeros_like(filtered_muscle_cols)
for i,muscle in enumerate(filtered_muscle_cols):
    n1,n2 = muscle.split('_')
    muscle_labels[i] = '%s_%s'%(n2[0].capitalize(), n1)

# def updatefig(*args):
while t<simulation_time:
    global counter,t
    # print(counter)
    print(t)
    time_window_inds = (flydf['t']>t)&(flydf['t']<=t+corr_window_size)
    state_mtrx = np.vstack([flydf[key][time_window_inds] for key in sorted_keys])
    # state_mtrx = np.vstack([flydf[key] for key in sorted_keys])
    # print(np.shape(state_mtrx))
    #state_mtrx = np.array(flydf.loc[time_window_inds,filtered_muscle_cols]).T
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
    # print('here')
    for n in G.nodes():
    # for n in G.nodes_iter():
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

    # edges= G.edges_iter()
    edges= G.edges()
    weights = [np.abs(G[e[0]][e[1]]['weight'])**2.5/1e5 for e in edges]
    # print(pos_dict)
    # nx.draw(G,ax = layout.axes['network_graph_layout'], pos = pos_dict,
    #         font_color = 'r', with_labels= False, width = weights,
    #         edge_color = colors, node_color = 'k', alpha = 0.1)
    plt.figure(1)
    # print(pos_dict)
    weights = np.array(weights)
    weights[np.isnan(weights)]=0.
    weights[np.isinf(weights)]=0.
    weights = list(np.array(weights).astype(str))
    # print(weights)
    # print(colors)
    # plt.gcf().canvas.flush_events()
    plt.clf()
    nx.draw(G, pos = pos_dict,
            font_color = 'k', with_labels= True, width = weights,
            edge_color = colors, node_color = 'k', alpha = 0.3)
    plt.show()
    # raw_input(' ')
    fig.savefig(filename+'.png')
    plt.pause(0.01)

    # layout.axes['network_graph_layout'].set_ybound(0,layout.axes['network_graph_layout'].h)
    # layout.axes['network_graph_layout'].set_xbound(0,layout.axes['network_graph_layout'].w)

    # print('here1')
    # layout.save(filename+'.svg',)
    # svg_data = open(filename+'.svg', 'r').read()
    # render_svg_to_png(svg_data,filename+'_'+str(counter)+'.png')
    # imported_image = plt.imread(filename+'.png',format='png')
    plt.imread(filename+'.png',format='png')
    # im.set_array(imported_image)
    plt.show()
    # print('here2')
    counter+=1
    # print(np.shape(imported_image))
    writer.grab_frame()
    # return im,
    t+=dt
writer.finish()
# ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
# plt.show()
