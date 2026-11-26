"""CPU synthetic contract tests; no detector, dataset, network, or training run."""
import unittest
import numpy as np
import torch
from refiner import (ROIRefiner, prepare_rois, fixed_bindings, utility_targets,
                     refinement_loss, gain_loss, apply_refinement, aligned_iou_giou)


def scalar_replacement_oracle(boxes0, boxes1, labels, scores, gt_boxes, gt_labels):
    """Deliberately slow independent scalar IoU/greedy implementation for tests.

    It constructs the complete changed output list for EACH scenario and
    rematches from scratch for EACH threshold. No optimised helper is used.
    Original order resolves score ties; later GT index resolves equal IoU.
    """
    b0, b1, gs = boxes0.tolist(), boxes1.tolist(), gt_boxes.tolist()
    labels, scores, gl = labels.tolist(), scores.tolist(), gt_labels.tolist()
    thresholds = [float(np.float32(x / 100)) for x in range(50, 100, 5)]
    order = sorted(range(len(b0)), key=lambda d: -scores[d])
    def iou(a, b):
        intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
        aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
        bb = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
        return float(np.float32(intersection / max(aa+bb-intersection, 1e-8)))
    def hits(boxes, threshold):
        matched = set()
        for d in order:
            best, best_iou = None, threshold
            for g in range(len(gs)):
                if g in matched or labels[d] != gl[g]: continue
                overlap = iou(boxes[d], gs[g])
                if overlap >= best_iou:
                    best, best_iou = g, overlap
            if best is not None: matched.add(best)
        return matched
    baseline = [hits(b0, threshold) for threshold in thresholds]
    new = torch.zeros((len(b0), 10), dtype=torch.long)
    lost = torch.zeros_like(new)
    for d in range(len(b0)):
        changed = b0.copy(); changed[d] = b1[d]
        for t, threshold in enumerate(thresholds):
            candidate = hits(changed, threshold)
            new[d, t] = len(candidate - baseline[t])
            lost[d, t] = len(baseline[t] - candidate)
    return new, lost


class RefinerContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def image(self, h=80, w=160):
        yy, xx = torch.meshgrid((torch.arange(h)+.5)/h,
                                (torch.arange(w)+.5)/w, indexing='ij')
        return torch.stack((xx, yy, (xx+yy)/2))

    def sample(self):
        image = self.image()
        boxes = torch.tensor([[64.,24.,80.,40.], [64.,24.,80.,40.], [0.,0.,120.,70.]])
        ids = torch.tensor([7,7,8])
        rois = prepare_rois(image, boxes, ids, input_hw=(80,160))
        return boxes, rois

    def test_query_reuse_continuous_coordinates_and_ineligible(self):
        image = self.image()
        boxes = torch.tensor([[64.,24.,79.,39.], [64.,24.,79.,39.], [0.,0.,120.,70.]])
        rois = prepare_rois(image, boxes, torch.tensor([7,7,8]), input_hw=(80,160))
        self.assertEqual(tuple(rois.crops.shape), (1,4,64,64))
        self.assertEqual(rois.det_to_roi.tolist(), [0,0,-1])
        self.assertTrue(torch.equal(rois.crops[:,3], torch.ones_like(rois.crops[:,3])))
        # Crop starts at 55.5,15.5 and first output centre is .25 pixel later.
        self.assertAlmostEqual(rois.crops[0,0,0,0].item(), 55.75/160, places=6)
        self.assertAlmostEqual(rois.crops[0,1,0,0].item(), 15.75/80, places=6)
        expected = torch.tensor([8.5,8.5,23.5,23.5])/32
        torch.testing.assert_close(rois.box_roi[0], expected)

    def test_raw_degraded_physical_range_and_masks_identical(self):
        image = self.image(96,160)
        boxes = torch.tensor([[1.,1.,20.,31.], [80.,40.,92.,55.]])
        q = torch.tensor([0,1])
        raw = prepare_rois(image, boxes, q, input_hw=(48,80), arm='raw')
        low = prepare_rois(image, boxes, q, input_hw=(48,80), arm='degraded')
        for name in ('roi_boxes','box_roi','det_to_roi','eligible'):
            self.assertTrue(torch.equal(getattr(raw,name), getattr(low,name)))
        self.assertTrue(torch.equal(raw.crops[:,3],low.crops[:,3]))
        self.assertGreater(int((raw.crops[:,3]==0).sum()),0)
        # Interior linear ramp is preserved under coordinate mapping.
        torch.testing.assert_close(raw.crops[1,:3,20:44,20:44],
                                   low.crops[1,:3,20:44,20:44],atol=2e-5,rtol=0)

    def test_inconsistent_query_and_wrong_transform_fail(self):
        b=torch.tensor([[0.,0.,10.,10.],[0.,0.,11.,10.]])
        with self.assertRaisesRegex(ValueError,'identical'):
            prepare_rois(self.image(),b,torch.tensor([2,2]),input_hw=(80,160))
        with self.assertRaisesRegex(ValueError,'resize'):
            prepare_rois(self.image(),b,torch.tensor([2,3]),input_hw=(80,160),
                         input_scale_xy=torch.tensor([.5,.5]))

    def test_native_degraded_tensor_is_used_without_second_resize(self):
        image=self.image(96,160);b=torch.tensor([[80.,40.,92.,55.]])
        q=torch.tensor([0]);native=torch.full((3,48,80),.25,requires_grad=True)
        raw=prepare_rois(image,b,q,input_hw=(48,80),arm='raw')
        low=prepare_rois(image,b,q,input_hw=(48,80),arm='degraded',degraded_image=native)
        self.assertTrue(torch.equal(raw.roi_boxes,low.roi_boxes))
        self.assertTrue(torch.equal(raw.crops[:,3],low.crops[:,3]))
        torch.testing.assert_close(low.crops[:,:3],torch.full_like(low.crops[:,:3],.25),atol=1e-6,rtol=0)
        self.assertFalse(low.crops.requires_grad)
        with self.assertRaisesRegex(ValueError,'degraded_image'):
            prepare_rois(image,b,q,input_hw=(48,80),arm='degraded',degraded_image=torch.zeros(3,47,80))

    def test_zero_init_no_base_grad_and_class_condition(self):
        b = torch.tensor([[64.,24.,79.,39.],[64.,24.,79.,39.],[0.,0.,120.,70.]],requires_grad=True)
        r = prepare_rois(self.image(),b,torch.tensor([7,7,8]),input_hw=(80,160))
        m=ROIRefiner(query_dim=256)
        h=torch.randn(3,256,requires_grad=True); labels=torch.tensor([0,1,2])
        out=m(r,b,labels,h)
        self.assertTrue(torch.equal(out['boxes1'],b))
        out['boxes1'].sum().backward()
        self.assertIsNone(b.grad); self.assertIsNone(h.grad)
        # Deterministically wire class 1 to one edge. Same query need not share b1.
        with torch.no_grad():
            m.geometry[0].weight.zero_();m.geometry[0].bias.zero_()
            m.geometry[0].weight[0,128+4+1]=1
            m.geometry[-1].weight.zero_();m.geometry[-1].bias.zero_()
            m.geometry[-1].weight[0,0]=1
        out=m(r,b,labels,h)
        self.assertEqual(out['boxes1'][0,0].item(),64.)
        self.assertGreater(out['boxes1'][1,0].item(),64.)
        self.assertTrue(torch.equal(out['boxes1'][2],b[2]))

    def test_edge_bound_and_positive_shape(self):
        b=torch.tensor([[64.,24.,79.,39.]])
        r=prepare_rois(self.image(),b,torch.tensor([7]),input_hw=(80,160))
        m=ROIRefiner(query_dim=0)
        with torch.no_grad():m.geometry[-1].bias.copy_(torch.tensor([100.,100.,-100.,-100.]))
        out=m(r,b,torch.tensor([0]));wh=b[:,2:]-b[:,:2]
        self.assertTrue(bool(((out['boxes1']-b).abs() <= .2*wh.repeat(1,2)+1e-6).all()))
        self.assertTrue(bool((out['boxes1'][:,2:]>out['boxes1'][:,:2]).all()))

    def test_true_075_success_has_priority_over_05_assignment(self):
        # High-score broad box could claim GT0 at .5, but low-score exact box
        # owns GT0 at .75. The .75 binding must be locked first.
        b=torch.tensor([[0.,0.,15.,10.],[0.,0.,10.,10.],[20.,0.,30.,10.]])
        g=torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.]])
        bind=fixed_bindings(b,torch.tensor([0,0,1]),torch.tensor([.9,.8,.7]),g,torch.tensor([0,1]))
        self.assertEqual(bind.det_to_gt.tolist(),[-1,0,1])
        self.assertEqual(bind.old_success.tolist(),[False,True,True])

    def test_multiclass_same_query_binds_separate_class_gts(self):
        b=torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]])
        bind=fixed_bindings(b,torch.tensor([0,1]),torch.tensor([.8,.7]),
                            b.clone(),torch.tensor([0,1]))
        self.assertEqual(bind.det_to_gt.tolist(),[0,1])

    def assert_utility_matches_oracle(self, b, p, labels, scores, g, gl, eligible=None):
        bind = fixed_bindings(b, labels, scores, g, gl)
        target = utility_targets(b, p, labels, g, gl, bind, eligible, scores=scores)
        new, lost = scalar_replacement_oracle(b, p, labels, scores, g, gl)
        self.assertTrue(torch.equal(target['new_count_by_threshold'], new))
        self.assertTrue(torch.equal(target['lost_count_by_threshold'], lost))
        self.assertTrue(torch.equal(target['measured_u'], (new-lost).float().mean(-1)))
        expected = torch.where(target['identity_valid'], target['measured_u'], torch.full_like(target['measured_u'], -1))
        self.assertTrue(torch.equal(target['target_y'], expected))
        self.assertTrue(bool(((new-lost).abs() <= 1).all()))
        for value in target.values(): self.assertFalse(value.requires_grad)
        return target

    def test_rematched_utility_detached_and_neighbor_switch_rejected(self):
        b=torch.tensor([[0.,0.,8.,10.],[20.,0.,30.,10.],[60.,0.,70.,10.]])
        g=torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.],[35.,0.,45.,10.]])
        labels=torch.tensor([0,0,1]);gl=torch.tensor([0,0,0])
        bind=fixed_bindings(b,labels,torch.tensor([.9,.8,.7]),g,gl)
        proposal=torch.tensor([[0.,0.,10.,10.],[35.,0.,45.,10.],[60.,0.,70.,10.]],requires_grad=True)
        t=self.assert_utility_matches_oracle(b,proposal,labels,torch.tensor([.9,.8,.7]),g,gl)
        self.assertAlmostEqual(t['local_fixed_gt_delta'][0].item(),.3,places=6)
        self.assertAlmostEqual(t['measured_u'][0].item(),.3,places=6)
        self.assertAlmostEqual(t['target_y'][0].item(),.3,places=6)
        self.assertEqual(t['target_y'][1].item(),-1.)
        self.assertEqual(t['target_y'][2].item(),-1.)
        self.assertEqual(t['identity_valid'].tolist(),[True,False,False])
        for value in t.values():self.assertFalse(value.requires_grad)
        self.assertTrue(torch.isnan(t['measured_delta'][2]))
        self.assertEqual(t['measured_u'][2].item(),0.)

    def test_duplicate_backup_breaks_local_delta_both_directions(self):
        g=torch.tensor([[0.,0.,10.,10.]])
        labels=torch.tensor([0,0]);scores=torch.tensor([.9,.8]);gl=torch.tensor([0])
        for old,new,local_expected in ((g[0],torch.tensor([1.,0.,11.,10.]),-.3),
                                       (torch.tensor([0.,0.,8.,10.]),g[0],.3)):
            b=torch.stack((old,g[0]));p=b.clone();p[0]=new
            t=self.assert_utility_matches_oracle(b,p,labels,scores,g,gl)
            self.assertAlmostEqual(t['local_fixed_gt_delta'][0].item(),local_expected,places=6)
            self.assertEqual(t['measured_u'][0].item(),0.)
            self.assertTrue(t['identity_valid'][0]);self.assertEqual(t['target_y'][0].item(),0.)

    def test_matching_chain_positive_measured_gain_can_have_invalid_identity(self):
        g=torch.tensor([[0.,0.,10.,10.],[3.,0.,13.,10.]])
        b=torch.stack((g[0],g[0]));p=b.clone();p[0]=g[1]
        t=self.assert_utility_matches_oracle(b,p,torch.tensor([0,0]),torch.tensor([.9,.8]),g,torch.tensor([0,0]))
        self.assertAlmostEqual(t['measured_u'][0].item(),.9,places=6)
        self.assertEqual(t['new_count_by_threshold'][0].tolist(),[0]+[1]*9)
        self.assertEqual(t['lost_count_by_threshold'][0].tolist(),[0]*10)
        self.assertFalse(t['identity_valid'][0]);self.assertEqual(t['target_y'][0].item(),-1.)

    def test_hit_identity_switch_has_both_gain_and_harm_with_zero_net(self):
        g=torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.]])
        t=self.assert_utility_matches_oracle(g[:1],g[1:],torch.tensor([0]),torch.tensor([.9]),g,torch.tensor([0,0]))
        self.assertEqual(t['new_count_by_threshold'][0].tolist(),[1]*10)
        self.assertEqual(t['lost_count_by_threshold'][0].tolist(),[1]*10)
        self.assertEqual(t['measured_u'][0].item(),0.)
        self.assertEqual(t['target_y'][0].item(),-1.)

    def test_each_scenario_replaces_only_one_item_not_all_proposals(self):
        g=torch.tensor([[0.,0.,10.,10.]])
        b=torch.tensor([[50.,0.,60.,10.],[80.,0.,90.,10.]])
        p=g.repeat(2,1)
        t=self.assert_utility_matches_oracle(b,p,torch.tensor([0,0]),torch.tensor([.9,.8]),g,torch.tensor([0]))
        # Either single replacement independently gains one; simultaneous
        # replacement would gain only one total and must not label the second 0.
        self.assertEqual(t['measured_u'].tolist(),[1.,1.])
        self.assertEqual(t['target_y'].tolist(),[-1.,-1.])

    def test_later_gt_iou_tie_and_stable_score_ties_multiclass(self):
        # First proposal exchanges the later GT hit for earlier GT at .50:.65.
        g=torch.tensor([[0.,0.,10.,10.],[4.,0.,14.,10.]])
        t=self.assert_utility_matches_oracle(torch.tensor([[2.,0.,12.,10.]]),g[:1],
            torch.tensor([0]),torch.tensor([.9]),g,torch.tensor([0,0]))
        self.assertEqual(t['new_count_by_threshold'][0,:4].tolist(),[1]*4)
        self.assertEqual(t['lost_count_by_threshold'][0,:4].tolist(),[1]*4)
        # Identical score ties retain original output order; classes never compete.
        g=torch.tensor([[0.,0.,10.,10.],[3.,0.,13.,10.],[0.,0.,10.,10.]])
        b=torch.stack((g[0],g[0],g[2]));p=b.clone();p[0]=g[1];p[2]+=20
        self.assert_utility_matches_oracle(b,p,torch.tensor([0,0,1]),torch.tensor([.8,.8,.8]),g,torch.tensor([0,0,1]))

    def test_threshold_edges_empty_outputs_gt_zero_and_ineligible_measured(self):
        empty=torch.empty(0,4);ids=torch.empty(0,dtype=torch.long)
        t=self.assert_utility_matches_oracle(empty,empty,ids,torch.empty(0),empty,ids)
        self.assertEqual(tuple(t['new_count_by_threshold'].shape),(0,10))
        b=torch.tensor([[0.,0.,10.,10.]])
        t=self.assert_utility_matches_oracle(b,b+1,torch.tensor([0]),torch.tensor([.5]),empty,ids)
        self.assertEqual(t['measured_u'].tolist(),[0.]);self.assertEqual(t['target_y'].tolist(),[-1.])
        # Exact 0.50 and 0.95 boundaries, and unbound-to-GT proposals, are measured.
        g=torch.tensor([[0.,0.,20.,20.]])
        for width in (10.,19.):
            p=torch.tensor([[0.,0.,width,20.]])
            t=self.assert_utility_matches_oracle(g+100,p,torch.tensor([0]),torch.tensor([.5]),g,torch.tensor([0]),torch.tensor([False]))
            self.assertEqual(int(t['new_count_by_threshold'].sum()),1 if width==10 else 10)
            self.assertFalse(t['identity_valid'][0]);self.assertEqual(t['target_y'][0].item(),-1.)

    def test_optimized_rematching_equals_independent_scalar_random_oracle(self):
        generator=torch.Generator().manual_seed(20261009)
        for case in range(24):
            g=torch.randint(0,25,(5,4),generator=generator).float()
            g[:,2:]=g[:,:2]+torch.randint(4,18,(5,2),generator=generator).float()
            b=g[torch.randint(0,5,(9,),generator=generator)].clone()
            b+=torch.randint(-3,4,(9,1),generator=generator).float()
            p=b+torch.randint(-4,5,(9,4),generator=generator).float()
            p[:,2:]=torch.maximum(p[:,2:],p[:,:2]+1)
            labels=torch.randint(0,3,(9,),generator=generator)
            gl=torch.randint(0,3,(5,),generator=generator)
            scores=torch.randint(0,3,(9,),generator=generator).float()/3
            eligible=torch.rand(9,generator=generator)>.25
            with self.subTest(case=case): self.assert_utility_matches_oracle(b,p,labels,scores,g,gl,eligible)

    def test_utility_requires_original_scores_and_rejects_nonfinite(self):
        b=torch.tensor([[0.,0.,10.,10.]]);l=torch.tensor([0]);s=torch.tensor([.9])
        bind=fixed_bindings(b,l,s,b,l)
        with self.assertRaises(TypeError): utility_targets(b,b,l,b,l,bind)
        with self.assertRaisesRegex(ValueError,'nonfinite score'):
            utility_targets(b,b,l,b,l,bind,scores=torch.tensor([float('nan')]))
        with self.assertRaisesRegex(ValueError,'nonfinite boxes'):
            utility_targets(b,b*float('nan'),l,b,l,bind,scores=s)

    def test_gate_stage_cannot_train_refiner(self):
        b=torch.tensor([[64.,24.,79.,39.]])
        r=prepare_rois(self.image(),b,torch.tensor([7]),input_hw=(80,160))
        m=ROIRefiner(query_dim=0);m.set_stage('gate')
        out=m(r,b,torch.tensor([0]));loss=gain_loss(out['gain'],{'target_y':torch.tensor([.3])},r.eligible)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in m.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in m.geometry.parameters()))
        self.assertTrue(any(p.grad is not None for p in m.gate.parameters()))

    def test_gain_bounded_and_threshold_one_guarantees_rejection(self):
        b=torch.tensor([[64.,24.,79.,39.]])
        r=prepare_rois(self.image(),b,torch.tensor([7]),input_hw=(80,160))
        m=ROIRefiner(query_dim=0);m.set_stage('gate')
        for bias in (1e6,-1e6):
            with torch.no_grad(): m.gate[-1].bias.fill_(bias)
            out=m(r,b,torch.tensor([0]))
            self.assertTrue(bool(((out['gain']>=-1)&(out['gain']<=1)).all()))
            result,accepted=apply_refinement(b,b+.1,out['gain'],r.eligible,threshold=1.)
            self.assertFalse(bool(accepted.any()));self.assertTrue(torch.equal(result,b))

    def test_balanced_gate_loss_present_groups_and_empty_are_finite(self):
        gain=torch.zeros(5,requires_grad=True)
        targets={'target_y':torch.tensor([1.,0.,-1.,-1.,-1.])}
        eligible=torch.ones(5,dtype=torch.bool)
        balanced=gain_loss(gain,targets,eligible)
        self.assertAlmostEqual(balanced.item(),1/3,places=6)
        plain=gain_loss(gain,targets,eligible,balance_groups=False)
        self.assertAlmostEqual(plain.item(),.4,places=6)
        balanced.backward();self.assertTrue(bool(torch.isfinite(gain.grad).all()))
        self.assertAlmostEqual(gain.grad[0].item(),-1/3,places=6)
        self.assertAlmostEqual(gain.grad[2].item(),1/9,places=6)
        for y in (torch.full((5,),-1.),torch.zeros(5),torch.ones(5)):
            loss=gain_loss(gain,{'target_y':y},eligible)
            self.assertTrue(bool(torch.isfinite(loss)))
        empty=gain_loss(gain,targets,torch.zeros(5,dtype=torch.bool))
        self.assertEqual(empty.item(),0.);self.assertTrue(empty.requires_grad)

    def test_rejected_and_ineligible_are_bitwise_unchanged_even_clip(self):
        b=torch.tensor([[-1.,0.,10.,10.],[90.,90.,110.,110.],[0.,0.,5.,5.]])
        p=b+1
        result,accepted=apply_refinement(b,p,torch.tensor([1.,-1.,1.]),torch.tensor([False,True,True]),clip_image_hw=(100,100))
        self.assertTrue(torch.equal(result[:2],b[:2]))
        self.assertEqual(accepted.tolist(),[False,False,True])
        self.assertTrue(torch.equal(result[2],p[2]))
        same, accepted = apply_refinement(b,b,torch.ones(3),torch.ones(3,dtype=torch.bool),
                                          policy='all',clip_image_hw=(100,100))
        self.assertTrue(torch.equal(same,b));self.assertFalse(bool(accepted.any()))

    def test_empty_and_loss_gradient(self):
        empty=torch.empty(0,4); ids=torch.empty(0,dtype=torch.long)
        r=prepare_rois(self.image(),empty,ids,input_hw=(80,160))
        m=ROIRefiner(query_dim=0);out=m(r,empty,ids)
        out['boxes1'].sum().backward()
        b=torch.tensor([[0.,0.,10.,10.]])
        bind=fixed_bindings(b,torch.tensor([0]),torch.tensor([.9]),b,torch.tensor([0]))
        p=torch.tensor([[1.,0.,10.,10.]],requires_grad=True)
        losses=refinement_loss(b,p,b,bind,torch.tensor([True]))
        self.assertGreater(losses['old_success_harm'].item(),0)
        losses['loss'].backward();self.assertIsNotNone(p.grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
