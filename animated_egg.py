import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import figurefirst as fifi
import flylib as flb
import cairo
import rsvg
import networkx as nx

fly_num = 1548
corr_window_size = 1

t=0

try:
    fly = flb.NetFly(fly_num,rootpath='/home/annie/imager/media/imager/FlyDataD/FlyDB/')
except(OSError):
    fly = flb.NetFly(fly_num)

flydf = fly.construct_dataframe()

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

def render_svg_to_png(svg_data,filename):
    # Render
    svg = rsvg.Handle(data=svg_data)
    img = cairo.ImageSurface(cairo.FORMAT_ARGB32,
      svg.props.width,
      svg.props.height)
    ctx = cairo.Context(img)
    svg.render_cairo(ctx)

    img.write_to_png(filename)

fig = plt.figure()
filename = 'f_egg'
layout = fifi.FigureLayout('graph_layout.svg')
im = plt.imshow(np.random.randn(10,10))

def updatefig(*args):

    #time_window_inds = (flydf['t']>t)&(flydf['t']<=t+corr_window_size)
    state_mtrx = np.vstack([flydf[key] for key in sorted_keys])
    #state_mtrx = np.array(flydf.loc[time_window_inds,filtered_muscle_cols]).T
    #Watch out for muscles that have no activity
    off_muscle_inds = (np.sum(state_mtrx,axis=1)==0.)
    # Set them to a very small amt of activity so nan's are not created
    state_mtrx[off_muscle_inds] = 1e-8
    centered_mtrx = state_mtrx - np.mean(state_mtrx,axis = 1)[:,None]
    std_mtrx = centered_mtrx/np.std(centered_mtrx,axis = 1)[:,None]
    cor_mtrx = np.dot(std_mtrx,std_mtrx.T)


    G = nx.Graph()
    for i,lbl1 in enumerate(filtered_muscle_cols):
        for j,lbl2 in enumerate(filtered_muscle_cols):
            G.add_edge(lbl1,lbl2,weight = cor_mtrx[i,j])

    c_ex = layout.pathspecs['excitatory'].mplkwargs()['edgecolor']
    c_in = layout.pathspecs['inhibitory'].mplkwargs()['edgecolor']
    colors = [{True:c_ex,False:c_in}[G[e[0]][e[1]]['weight']>0.] for e in G.edges()]

    h = float(layout.layout_uh)
    pos_dict = {}
    for n in G.nodes():
        n1, n2 = n.split('_')
        n_s = '%s_%s'%(n2[0].capitalize(), n1)
        #n_s = '%s_%s'%(n[0][0].capitalize(),n[1])
        cx = float(layout.pathspecs[n_s]['cx'])
        cy = h-float(layout.pathspecs[n_s]['cy'])
        try:
            if 'transform' in layout.pathspecs[n_s].keys():
                t1 = fifi.svg_to_axes.parse_transform(layout.pathspecs[n_s]['transform'])
                p = np.dot(t1,np.array([cx,cy,1]))
                pos_dict[n] = (p[0],p[1])
            else:
                pos_dict[n]  = (cx,cy)
        except KeyError:
            print n

    edges= G.edges
    weights = [np.abs(G[e[0]][e[1]]['weight'])**2.6/100000000000. for e in edges]
    nx.draw(G,ax = layout.axes['network_graph_layout'], pos = pos_dict,
            font_color = 'r', with_labels= False, width = weights,
            edge_color = colors, node_color = 'k', alpha = 0.1)

    layout.axes['network_graph_layout'].set_ybound(0,layout.axes['network_graph_layout'].h)
    layout.axes['network_graph_layout'].set_xbound(0,layout.axes['network_graph_layout'].w)

    layout.save(filename+'.svg',)
    svg_data = open(filename+'.svg', 'r').read()
    render_svg_to_png(svg_data,filename+'.png')
    imported_image = plt.imread(filename+'.png',format='png')
    im.set_array(imported_image)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
