def strip_bare(ax):
    ax.set_xticks([])
    ax.set_xticklabels('')
    ax.set_yticks([])
    ax.set_yticklabels('')

def strip_ticks(ax):
   ax.tick_params(axis=u'both', which=u'both',length=0)