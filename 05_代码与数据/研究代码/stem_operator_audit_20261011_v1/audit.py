"""Read-only algebra and operator examples, not a learned-model efficacy test."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
STEM=ROOT/'英文小论文/code/original_stem12_20261011_v1/original_stem.py'
CONTROL=ROOT/'英文小论文/code/stem_component12_20261011_v1/stem_controls.py'
STEM_SHA='ca9884aebe8fe3b2bbc1f3f36ea2fed17f2c817b464a2581f22c2f4d3ffb2c74'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def box3_zero_pad(x):
    x=np.asarray(x,dtype=np.float64);a=np.pad(x,1,mode='constant')
    return sum(a[i:i+x.shape[0],j:j+x.shape[1]] for i in range(3) for j in range(3))/9

def exact_half_area(x):
    # For an even H,W, half-size bilinear align_corners=False is this 2x2 mean.
    # This function does not include the learned high_freq_enhance operations.
    h,w=x.shape;assert h%2==w%2==0
    return x.reshape(h//2,2,w//2,2).mean(axis=(1,3))

def receptive_field(kernels,strides):
    r,j=1,1
    for k,s in zip(kernels,strides):r+=(k-1)*j;j*=s
    return dict(receptive_field_side_in_input_grid=r,output_stride=j)

def audit(output):
    assert sha(STEM)==STEM_SHA
    stem=ast.parse(STEM.read_text(encoding='utf8'));control=ast.parse(CONTROL.read_text(encoding='utf8'))
    f=next(n for n in ast.walk(stem) if isinstance(n,ast.FunctionDef) and n.name=='forward' and any(isinstance(x,ast.Name) and x.id=='weighted_feat' for x in ast.walk(n)))
    assigns={x.targets[0].id:ast.unparse(x.value) for x in f.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name)}
    assert assigns['weighted_feat']=='concat_feat * spatial_att + concat_feat'
    assert assigns['blurred']=='F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)'
    assert assigns['high_freq']=='F.interpolate(high_freq, size=(main.shape[2], main.shape[3]), mode=\'bilinear\', align_corners=False)'
    calls=[ast.unparse(x) for x in ast.walk(control) if isinstance(x,ast.Assign)]
    assert 'self.fusion[1].conv.stride = (2, 2)' in calls
    assert 'self.final_downsample.conv.stride = (1, 1)' in calls
    n=64;constant=np.ones((n,n));dc=constant-box3_zero_pad(constant)
    assert np.array_equal(dc[1:-1,1:-1],np.zeros((n-2,n-2)))
    assert np.isclose(dc[0,0],5/9) and np.isclose(dc[0,2],1/3)
    signals={}
    yy,xx=np.indices((n,n))
    for name,x in dict(checkerboard=(-1.)**(xx+yy),horizontal_nyquist=(-1.)**xx).items():
        high=x-box3_zero_pad(x);pooled=exact_half_area(high)
        interior=high[2:-2,2:-2];after=pooled[1:-1,1:-1]
        assert np.allclose(after,0)
        signals[name]=dict(input_interior_RMS=float(np.sqrt(np.mean(x[2:-2,2:-2]**2))),
            extracted_interior_RMS=float(np.sqrt(np.mean(interior**2))),
            half_res_without_learned_block_interior_RMS=float(np.sqrt(np.mean(after**2))))
    full=receptive_field((3,3),(1,2));early=receptive_field((3,3),(2,1))
    assert full['receptive_field_side_in_input_grid']==5 and early['receptive_field_side_in_input_grid']==7
    alternating=np.array([[-1.,1.],[-1.,1.]])
    pre=float(np.maximum(alternating,0).mean());post=float(np.maximum(alternating.mean(),0))
    assert pre==.5 and post==0
    p=ROOT/'英文小论文/文献_相关原始论文_20261004/187_TinyViM_ICCV2025.pdf'
    t=p.parent/'_text/187_TinyViM_ICCV2025.txt'
    assert sha(p)=='7c1ade779ca4b7d23b0e9a99028669f39635be4ffc8c0e24434b2b0996d6999d'
    assert sha(t)=='4fc63e04b9f3b7728547e597df2bb27e078b44e65ace1536bd30a9d59f26f695'
    out=dict(status='SOURCE_PINNED_ALGEBRA_AND_NUMPY_EXAMPLES_ONLY',
        files_sha256={str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in (STEM,CONTROL,p,t,Path(__file__))},
        attention=dict(expression=assigns['weighted_feat'],real_arithmetic_gain='1 < 1+sigmoid(z) < 2; finite precision may reach the endpoints',
            implication='Direct gain cannot attenuate a feature magnitude. Relative target/background reweighting can still change, and downstream convolution/BN can suppress features; neither suppression nor task benefit is proved here.'),
        raw_residual=dict(infinite_interior_frequency_response='1 - (1+2*cos(wx))*(1+2*cos(wy))/9',
            constant_input_interior_max_abs=float(abs(dc[1:-1,1:-1]).max()),corner_value=float(dc[0,0]),edge_value=float(dc[0,2]),
            boundary='Zero padding and default count_include_pad=True create residual at a constant image boundary. This is not evidence of a detection error.',
            half_sampling_examples=signals,
            limitation='Synthetic preprocessing/half-res examples omit Conv/BN/ReLU and do not reproduce the complete trained high-frequency branch. They refute automatic all-high-frequency preservation, not its learned efficacy.'),
        downsample_control=dict(full_after_attention=full,early_after_attention=early,
            grid='Concatenated half-resolution feature grid; this does not represent the entire Stem receptive field, whose attention and three inputs have other support.',
            interpretation='This comparison jointly changes decimation location and subsequent receptive field. A conditional AP difference cannot uniquely identify aliasing/detail preservation.'),
        detail_pool_order=dict(toy_average_after_relu=pre,toy_relu_after_average=post,
            limitation='One nonlinear toy example establishes noncommutation only. Actual two learned Conv/BN/activation/LAB blocks also change effective receptive field and normalization distribution when moved after input pooling.'),
        paper=dict(title='TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba',venue='ICCV2025 main conference',
            primary_page='https://openaccess.thecvf.com/content/ICCV2025/html/Ma_TinyViM_Frequency_Decoupling_for_Tiny_Hybrid_Vision_Mamba_ICCV_2025_paper.html',
            new_reading_scope='Abstract; section1, Figure2 caption and related text; section5.4, Tables3-6, Figure4 caption; section5.2/Table7 detection context. Source code and supplements not examined; no new PDF downloaded.',
            lesson='Compare representations before/after the questioned block, formulate a frequency allocation hypothesis, then compare alternative allocations rather than treating a high-frequency name as proof.',
            transfer_limit='Their observed spectral preference is for a Conv-Mamba hybrid. HGNetv2 does not thereby inherit this diagnosis. Their classification allocation ablations and COCO Mask R-CNN results do not establish VisDrone APs gains or novelty of this Stem.'),
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,trained_network_feature_arrays_analyzed=0,
        fixed12_efficacy_conclusion='PENDING',causal_mechanism_proven=False)
    assert 'torch' not in sys.modules
    with Path(output).open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(out['status'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();audit(a.output)
