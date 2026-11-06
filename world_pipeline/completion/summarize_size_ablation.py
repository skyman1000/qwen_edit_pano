"""Print the four saved size/text ablation results; CPU only, no model loading."""
from pathlib import Path

import numpy as np

from .geometry import read
from ..common import ROOT


def main():
    sample_id = 'X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c'
    for run in ('completion_indoor_val3_run001', 'completion_indoor_notext_run001',
                'completion_indoor_halftext_run001', 'completion_indoor_halfnotext_run001'):
        directory = ROOT / 'qwen_edit_pano/outputs' / run / sample_id
        config = read(directory / 'config.json')
        summary = read(directory / 'seed000/summary.json')
        print(f'\n{run}: size={config["size_convention"]}, text={config["text"]!r}')
        print(f'observed={summary["observed"]}, generated={summary["generated"]}, '
              f'ended={summary["ended"]}')
        print('category                   full_AABB_xyz_m          bottom_above_assumed_floor_m')
        for obj in read(directory / 'seed000/layout.json'):
            if obj['observed']:
                continue
            corners = np.asarray(obj['corners_local'])
            size = np.ptp(corners, axis=0)
            # Scene rotation is yaw-only, so vertical offset alone restores scene y.
            bottom = corners[:, 1].min() + config['camera_height']
            dims = ', '.join(f'{v:.3f}' for v in size)
            print(f'{obj["category"]:26s} [{dims}]  {bottom:+.3f}')
    print('\nBottom is relative to the assumed scene floor, not measured ground truth. '
          'Wall-mounted and supported objects need not touch that floor.')


if __name__ == '__main__':
    main()
