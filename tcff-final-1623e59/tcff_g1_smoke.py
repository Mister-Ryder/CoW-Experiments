"""One-case G1 invariance, intervention, and deterministic-corruption tests."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from topcow_dataset import TopCoWTestDataset
from topcow_eval_official_all import evaluate_single_case, load_model


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def main(args):
    root = Path(args.project_root).resolve()
    device = torch.device(args.device)
    model = load_model(str(root / 'Runs_foundation' / 'sam_tgf_v3'), 'tgf', device)
    dataset = TopCoWTestDataset(str(root / 'crown_processed'), 'test')
    target = dataset[0]
    donor = dataset[1]
    x = target['image'].unsqueeze(0).float().to(device)
    donor_x = donor['image'].unsqueeze(0).float().to(device)
    gt = target['multiclass'].numpy()

    def forward(variant='full', corruption=None, instrument=False,
                donor_features=None, gate_alpha=None):
        model.set_rebuttal_runtime(
            trust_variant=variant,
            corruption=corruption or {'type': 'none', 'severity': 0, 'seed': 0},
            instrumentation=instrument,
            donor_features=donor_features,
            gate_alpha=gate_alpha)
        result = model(x)[0]
        assert torch.isfinite(result).all()
        return result

    clean = forward()
    instrumented = forward(instrument=True)
    signals = model.get_rebuttal_signals()
    hook_diff = float((clean - instrumented).abs().max())
    assert hook_diff < 1e-6
    assert all(torch.isfinite(value).all()
               for level in signals.values() for value in level.values())

    reference = {}
    reference_path = root / 'results_tcff_rebuttal' / '01_configs' / 'preinstrument_reference.json'
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        assert tensor_hash(clean) == reference['seg_sha256']

    corruption_results = {}
    specs = [
        ('gaussian', 0.10),
        ('translation', 2.0),
        ('channel_dropout', 0.25),
        ('attenuation', 0.50),
    ]
    for kind, severity in specs:
        cfg = {'type': kind, 'severity': severity, 'seed': 123}
        first = forward(corruption=cfg, instrument=True)
        second = forward(corruption=cfg, instrument=True)
        repeat_diff = float((first - second).abs().max())
        clean_diff = float((first - clean).abs().max())
        assert repeat_diff == 0.0
        assert clean_diff > 0.0
        corruption_results[kind] = {
            'severity': severity,
            'repeat_max_abs_diff': repeat_diff,
            'clean_max_abs_diff': clean_diff,
            'prediction_sha256': tensor_hash(first.argmax(1)),
        }

    # Capture donor auxiliary tensors without selecting the donor by labels/results.
    model.set_rebuttal_runtime(
        trust_variant='full',
        corruption={'type': 'none', 'severity': 0, 'seed': 0},
        instrumentation=True,
        donor_features=None,
        gate_alpha=None)
    model(donor_x)
    donor_features = {
        level: sig['A_clean'] for level, sig in model.get_rebuttal_signals().items()}
    donor_cfg = {'type': 'donor_mix', 'severity': 0.25, 'seed': 123}
    donor_first = forward(corruption=donor_cfg, instrument=True,
                          donor_features=donor_features)
    donor_second = forward(corruption=donor_cfg, instrument=True,
                           donor_features=donor_features)
    assert float((donor_first - donor_second).abs().max()) == 0.0
    assert float((donor_first - clean).abs().max()) > 0.0
    corruption_results['donor_mix'] = {
        'severity': 0.25,
        'donor_case': donor['case_name'],
        'repeat_max_abs_diff': float((donor_first - donor_second).abs().max()),
        'clean_max_abs_diff': float((donor_first - clean).abs().max()),
    }

    variant_results = {}
    variant_outputs = {}
    for variant in ('full', 'product', 'u_only', 'g_only', 'residual',
                    'no_aux', 'additive'):
        output = forward(variant=variant, instrument=True)
        variant_outputs[variant] = output
        pred = output[0].argmax(0).cpu().numpy()
        metrics = evaluate_single_case(pred, gt)
        variant_results[variant] = {
            'binary_dice': metrics['binary_dice'],
            'class_avg_dice': metrics['class_avg_dice'],
            'output_sha256': tensor_hash(output),
        }
        if variant == 'product':
            residuals = []
            for level, signals in model.get_rebuttal_signals().items():
                residual = float((signals['A_impl'] -
                                  signals['T_formula']).abs().max())
                assert residual < 1e-6, level
                residuals.append(residual)
            variant_results[variant][
                'max_abs_A_impl_minus_T_formula'] = max(residuals)

    gate_dose = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        output = forward(variant='full', instrument=True, gate_alpha=alpha)
        pred = output[0].argmax(0).cpu().numpy()
        metrics = evaluate_single_case(pred, gt)
        gate_dose.append({
            'alpha': alpha,
            'binary_dice': metrics['binary_dice'],
            'class_avg_dice': metrics['class_avg_dice'],
            'output_sha256': tensor_hash(output),
        })
        if alpha == 0.0:
            assert float((output - variant_outputs['no_aux']).abs().max()) < 1e-6
        if alpha == 1.0:
            assert float((output - variant_outputs['residual']).abs().max()) < 1e-6

    result = {
        'status': 'success',
        'case_id': target['case_name'],
        'reference': reference,
        'clean_output_sha256': tensor_hash(clean),
        'hook_output_sha256': tensor_hash(instrumented),
        'hook_max_abs_diff': hook_diff,
        'signal_shapes': {
            level: {name: list(value.shape) for name, value in sig.items()}
            for level, sig in signals.items()
        },
        'corruptions': corruption_results,
        'variants': variant_results,
        'gate_dose': gate_dose,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'status': result['status'],
        'hook_max_abs_diff': hook_diff,
        'corruptions': list(corruption_results),
        'variants': list(variant_results),
        'gate_doses': len(gate_dose),
    }, sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    p.add_argument('--project_root', default=str(here))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', default=str(
        here / 'results_tcff_rebuttal' / '01_configs' / 'g1_smoke_results.json'))
    main(p.parse_args())
