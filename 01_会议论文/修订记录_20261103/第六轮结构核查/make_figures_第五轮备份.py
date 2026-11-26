from pathlib import Path
from xml.sax.saxutils import escape

OUT=Path(__file__).resolve().parent/'figures'
OUT.mkdir(exist_ok=True)
BLUE='#eaf1f7'; GREEN='#edf3ec'; ORANGE='#fff0df'; GRAY='#f4f4f4'
class Canvas:
    def __init__(self,w,h):self.w=w;self.h=h;self.parts=[]
    def text(self,x,y,s,ha='left',va='center',fontsize=10,**kw):
        lines=s.split('\n');size=fontsize*1.22*100/72;lh=size*1.2
        yy=(self.h-y)*100-(len(lines)-1)*lh/2
        anchor={'center':'middle','left':'start','right':'end'}[ha]
        for i,t in enumerate(lines):self.parts.append(f'<text x="{x*100}" y="{yy+i*lh}" dominant-baseline="central" text-anchor="{anchor}" font-family="Arial" font-size="{size}">{escape(t)}</text>')
def canvas(w,h):
    a=Canvas(w,h);return a,a
def box(a,x,y,w,h,label,c=BLUE,size=10):
    a.parts.append(f'<rect x="{x*100}" y="{(a.h-y-h)*100}" width="{w*100}" height="{h*100}" fill="{c}" stroke="#333333" stroke-width="1.2"/>')
    a.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)
def arrow(a,p,q,style='-',c='#333333'):
    dash=' stroke-dasharray="6,4"' if style=='--' else ''
    a.parts.append(f'<line x1="{p[0]*100}" y1="{(a.h-p[1])*100}" x2="{q[0]*100}" y2="{(a.h-q[1])*100}" stroke="{c}" stroke-width="1.3" marker-end="url(#arr)"{dash}/>')
def line(a,pts,style='-'):
    dash=' stroke-dasharray="6,4"' if style=='--' else ''
    ps=' '.join(f'{x*100},{(a.h-y)*100}' for x,y in pts)
    a.parts.append(f'<polyline points="{ps}" fill="none" stroke="#333333" stroke-width="1.3"{dash}/>')
def save(f,name):
    head=f'<svg xmlns="http://www.w3.org/2000/svg" width="{f.w*100}" height="{f.h*100}" viewBox="0 0 {f.w*100} {f.h*100}"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#333333"/></marker></defs><rect width="100%" height="100%" fill="white"/>'
    (OUT/f'{name}.svg').write_text(head+''.join(f.parts)+'</svg>',encoding='utf-8')

f,a=canvas(10,3.9)
items=[('Input\n3 × 640 × 640',3.05),('MPS\n16 × 160 × 160',2.35),('Stage 1 · ESE\n64 × 160 × 160',1.65),('Stage 2 · SimAM\n256 × 80 × 80',.95),('Stage 3 · MSCA-Lite × 2\n512 × 40 × 40',.25)]
for lab,y in items:box(a,.2,y,2.7,.55,lab)
for i in range(4):arrow(a,(1.55,items[i][1]),(1.55,items[i+1][1]+.55))
box(a,3.35,.25,2.7,.55,'Stage 4 · MSCA-Dilated\n1024 × 20 × 20')
arrow(a,(2.9,.52),(3.35,.52))
box(a,6.55,1.3,3.1,1.1,'Hybrid encoder\n n: strides 16, 32; width 128\n s: strides 8, 16, 32; width 256',GREEN,10)
arrow(a,(6.05,.52),(6.7,1.3));line(a,[(2.7,.8),(2.7,.87),(6.3,.87),(6.3,1.55)]);arrow(a,(6.3,1.55),(6.55,1.55))
line(a,[(2.9,1.22),(5.95,1.22),(5.95,2.12)],'--');arrow(a,(5.95,2.12),(6.55,2.12),'--')
a.text(4.45,1.43,'Stage 2 output: s only',fontsize=9,ha='center')
box(a,6.55,2.9,3.1,.65,'D-FINE decoder → boxes and scores',ORANGE,10);arrow(a,(8.1,2.4),(8.1,2.9))
a.text(4.65,3.15,'HGNetv2-B0 in both scales\nBackbone stride = 4 → 8 → 16 → 32',ha='center',fontsize=10)
save(f,'fig1_architecture')

f,a=canvas(10,4.3)
box(a,.1,1.8,.9,.6,'Input\n3 × H × W',GRAY,9)
box(a,1.4,3.05,3.1,.75,'Detail: 3×3 s1 → 3×3 s1\n8 channels → AvgPool 2×2 s2',BLUE,9)
box(a,1.4,1.75,3.1,.75,'Main: 3×3 s2 → 3×3 s1\n16 channels',BLUE,9)
box(a,1.4,.45,3.1,.75,'High-pass: X − AvgPool 3×3 s1\nConv 1×1 → DWConv 3×3; 4 channels\nResize to H/2 × W/2',BLUE,9)
for y in [3.425,2.125,.825]:line(a,[(1,2.1),(1.2,2.1),(1.2,y)]);arrow(a,(1.2,y),(1.4,y))
box(a,4.9,1.7,1,.85,'Concat\n28 channels',GREEN,9)
for y in [3.425,2.125,.825]:line(a,[(4.5,y),(4.7,y),(4.7,2.125)]);arrow(a,(4.7,2.125),(4.9,2.125))
box(a,6.25,2.95,3.5,.8,'Spatial gate A(Z)\n1×1 (28→2) → BN → activation\n7×7 (2→1) → sigmoid',ORANGE,9)
line(a,[(5.4,2.55),(5.4,3.35)]);arrow(a,(5.4,3.35),(6.25,3.35))
box(a,6.25,1.7,3.5,.8,'Residual weighting: Z · (1 + A(Z))\nFusion: Conv 1×1 (28→32)\n→ Conv 3×3 s1 (32→16)',GREEN,9)
arrow(a,(5.9,2.125),(6.25,2.125));arrow(a,(8,2.95),(8,2.5))
box(a,6.25,.45,3.5,.65,'Conv 3×3 s2 (16→16)\nOutput: 16 × H/4 × W/4',BLUE,9);arrow(a,(8,1.7),(8,1.1))
a.text(.2,.15,'Conv blocks include BN and the configured activation. DWConv denotes depthwise convolution.',fontsize=8.6)
save(f,'fig2_mps')

f,a=canvas(10,2.3)
stages=[('Stage 1\nESE\n64 channels · stride 4',.1),('Stage 2\nSimAM\n256 channels · stride 8',2.6),('Stage 3\nMSCA-Lite × 2 blocks\n512 channels · stride 16',5.1),('Stage 4\nMSCA-Dilated\n1024 channels · stride 32',7.6)]
for lab,x in stages:box(a,x,.85,2.25,1,lab,BLUE,9.5)
for x in [2.35,4.85,7.35]:arrow(a,(x,1.35),(x+.25,1.35))
a.text(5,.36,'Each gate follows the HG-block aggregation projection; stage boundaries include the backbone convolutions.',ha='center',fontsize=9)
save(f,'fig3_stages')

f,a=canvas(10,2.9)
box(a,.1,1.05,1.1,.65,'Input F\nC × H × W',GRAY,9)
box(a,1.6,1.0,2.4,.85,'Identity\nAvgPool 2×2 → bilinear up\nAvgPool 4×4 → bilinear up',BLUE,9)
box(a,4.4,1.0,1.5,.85,'Mean of\nthree maps\nFms',GREEN,9)
box(a,6.3,1,2.2,.85,'Energy gate G(Fms)\nspatial mean / variance\nλ = 0.0001 → sigmoid',ORANGE,9)
box(a,8.95,1.05,.9,.65,'Output\nF · G',GREEN,9)
for x,y in [(1.2,1.6),(4,4.4),(5.9,6.3),(8.5,8.95)]:arrow(a,(x,1.425),(y,1.425))
line(a,[(.65,1.7),(.65,2.4),(9.4,2.4)]);arrow(a,(9.4,2.4),(9.4,1.7))
a.text(5,2.55,'Original feature F supplies the multiplicative value path',ha='center',fontsize=9)
a.text(5,.35,'No global pooling before the energy gate. All three maps retain C × H × W after alignment.',ha='center',fontsize=9)
save(f,'fig4_lite')

f,a=canvas(10,4.2)
box(a,.1,1.65,1,.65,'Input F',GRAY,10)
for y,d in [(2.65,1),(1.65,2),(.65,4)]:
    box(a,1.5,y,3,.7,f'1×1 (C→C/8) → BN → ReLU\n3×3 d{d} (C/8→C), groups C/8 → BN',BLUE,8.7)
    line(a,[(1.1,1.975),(1.3,1.975),(1.3,y+.35)]);arrow(a,(1.3,y+.35),(1.5,y+.35))
    line(a,[(4.5,y+.35),(4.75,y+.35),(4.75,1.975)])
box(a,5,1.65,1.5,.7,'Concat → Conv 1×1\n3C → C',GREEN,9);arrow(a,(4.75,1.975),(5,1.975))
box(a,6.85,1.65,1.45,.7,'Spatial gate\nAs = G(Cctx)',ORANGE,9);arrow(a,(6.5,1.975),(6.85,1.975))
box(a,5,3.05,3.3,.7,'Channel gate Ac(F): GAP → 1×1\nC→C/8 → ReLU → 1×1 → sigmoid',ORANGE,8.8)
line(a,[(.6,2.3),(.6,3.4)]);arrow(a,(.6,3.4),(5,3.4))
box(a,8.7,1.65,1.2,.7,'Output\nF · As · Ac',GREEN,9);arrow(a,(8.3,1.975),(8.7,1.975))
line(a,[(8.3,3.4),(9.3,3.4)]);arrow(a,(9.3,3.4),(9.3,2.35))
line(a,[(.6,1.65),(.6,.25),(9.3,.25)]);arrow(a,(9.3,.25),(9.3,1.65))
a.text(6.8,.83,'groups = C/8; 8 outputs per input group\nEnergy gate uses per-channel spatial statistics',ha='center',fontsize=8.8)
save(f,'fig5_dilated')
