"""Reuse the unchanged Qwen-Pano profiles implementation."""
from qwen_pano.profiles import REPO_PANORAMA, lr_multiplier


def apply_profile(args, parser):
    from qwen_pano.profiles import apply_profile as original
    if args.profile != "repo-panorama" or args.perspective_manifest:
        parser.error("Edit migration supports repo-panorama only, without reference or perspective images")
    original(args, parser)
