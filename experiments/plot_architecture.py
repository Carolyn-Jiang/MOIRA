"""Python vector redraws of the three existing appendix workflow diagrams.

The nodes preserve the manuscript's workflow and decision boundaries; these
schematics contain no empirical values and do not imply additional experiments.
"""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def generate_architecture(output):
    OUT=Path(output);OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'qa').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})

    def canvas():
     fig,ax=plt.subplots(figsize=(7.0,3.4));fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02);ax.set(xlim=(0,100),ylim=(-2,50));ax.axis('off');return fig,ax

    def box(ax,x,y,w,h,title,body,color='#DCEAF1'):
     ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.2,rounding_size=.6',lw=.6,edgecolor='#758A98',facecolor=color))
     ax.text(x+w/2,y+h-2.6,title,ha='center',va='top',weight='bold',fontsize=8)
     body='\n'.join(textwrap.fill(line,width=int(w*1.12),break_long_words=False,break_on_hyphens=False) for line in body.split('\n'))
     ax.text(x+w/2,y+h/2-1,body,ha='center',va='center',fontsize=7.5,linespacing=1.45)

    def arrow(ax,a,b,label=None):
     ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.7,color='#526775'))
     if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+1.2,label,ha='center',va='bottom',fontsize=7)

    def save(fig,name):
     from experiments.figure_alignment import require_matplotlib_panel_alignment
     require_matplotlib_panel_alignment(fig,json_out=str(OUT/'qa'/f'{name}.alignment.json'),strict=True)
     fig.savefig(OUT/f'{name}.pdf')
     fig.savefig(OUT/f'{name}.svg')
     svg_path = OUT/f'{name}.svg'
     svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
     fig.savefig(OUT/f'{name}.png',dpi=600)
     plt.close(fig)

    fig,ax=canvas()
    box(ax,1,28,17,20,'Current policy','Base prompt\nAccepted principles\nHistorical training data')
    box(ax,21,28,17,20,'Training replay','Daily actions\nRealized reward\nSame training window','#E4EBDD')
    box(ax,41,28,17,20,'Counterfactuals','All-action outcomes\nBest alternative\nError and regret','#F1E9D6')
    box(ax,61,28,17,20,'Reflection','Extract 1–3\ntransferable principles\nAvoid ticker-specific rules','#E7E0EF')
    box(ax,81,28,18,20,'Candidate policy','Base prompt +\nretained principles +\nnew principles')
    for a,b in [(18,21),(38,41),(58,61),(78,81)]:arrow(ax,(a,38),(b,38))
    box(ax,70,5,28,17,'Reward validation','Replay the same training window\nCompare candidate and current reward','#E4EBDD')
    box(ax,36,5,29,17,'Accept or retain','Higher reward: accept candidate\nOtherwise: retain current policy','#DCEAF1')
    box(ax,1,5,30,17,'Next round / early stopping','Continue bounded consolidation\nFreeze global policy during testing','#E7E0EF')
    arrow(ax,(90,28),(90,22));arrow(ax,(70,13.5),(65,13.5));arrow(ax,(36,13.5),(31,13.5))
    save(fig,'belief_update_readable')

    fig,ax=canvas()
    box(ax,1,27,18,21,'1. Candidate pairs','Ten-stock universe\n45 unordered pairs\nCausal 10-year prices')
    box(ax,21.5,27,18,21,'2. Statistical rank','Engle–Granger test\nCointegration p-value\nRank candidate pairs','#E4EBDD')
    box(ax,42,27,17,21,'3. Plausibility','ADF and half-life\n2 ≤ half-life ≤ 60\nScreen relationships','#F1E9D6')
    box(ax,61.5,27,17,21,'4. Candidate pool','Up to five pairs\nLowest-p-value fallback\nif none qualifies','#E7E0EF')
    box(ax,81,27,18,21,'5. LLM selection','Pair statistics\nNews / company context\nDerived pair memory')
    for a,b in [(19,21.5),(39.5,42),(59,61.5),(78.5,81)]:arrow(ax,(a,37),(b,37))
    box(ax,1,4,46,17,'Hypothesis selection','Choose the relationship supported by current evidence.\nRetain compact derived state, not raw observations.','#E7E0EF')
    box(ax,52,4,47,17,'Default selection schedule','Initially, then only at 23-session validation boundaries\nwhen the preceding window contains Break.','#E4EBDD')
    arrow(ax,(90,27),(90,21));arrow(ax,(52,12.5),(47,12.5))
    save(fig,'pair_select_readable')

    fig,ax=canvas()
    box(ax,1,29,18,19,'Active hypothesis','Held pair\nRelationship confidence')
    box(ax,22,29,36,19,'Daily relationship monitoring','Correlation · half-life · ADF · cointegration\nSpread z-score · spread volatility\nIndependent news-relation evidence','#E4EBDD')
    box(ax,61,29,38,19,'Confidence state','Statistical score: Healthy ≤ 3; Warning 3–5; Break ≥ 5\nNews evidence can raise the final state.\nBreak never forces an immediate trade.','#F1E9D6')
    arrow(ax,(19,38),(22,38));arrow(ax,(58,38),(61,38))
    box(ax,1,3,25,18,'Validation boundary','Every 23 sessions\nInspect the preceding cycle\nWas any Break recorded?','#E7E0EF')
    box(ax,31,3,30,18,'No Break: retain','Continue the current pair\nContinue confidence monitoring','#E4EBDD')
    box(ax,66,3,33,18,'Break: revise hypothesis','Trigger the pair-selection module\nSelect a new pair\nReset relationship confidence','#F0DEDC')
    arrow(ax,(13.5,29),(13.5,21));arrow(ax,(26,13),(31,13),'No');ax.plot([13.5,13.5,82.5],[3,0,0],color='#526775',lw=.7);arrow(ax,(82.5,0),(82.5,3));ax.text(49,-1.7,'Yes',fontsize=7,ha='center')
    save(fig,'lifecycle_readable')
    print('Wrote three appendix architecture figures.')
