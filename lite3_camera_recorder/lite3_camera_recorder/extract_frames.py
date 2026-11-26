"""Export numbered PNGs from a recorded video for annotation."""

import argparse
from pathlib import Path

import cv2


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path, help='recorded MP4 file')
    parser.add_argument('--every', type=int, default=1, metavar='N',
                        help='export the first frame and then every Nth frame (default 1)')
    parser.add_argument('--output-dir', type=Path,
                        help='new output directory (default VIDEO_frames)')
    options = parser.parse_args(args)
    if options.every < 1:
        parser.error('--every must be at least 1')

    video = options.video.expanduser()
    output_dir = options.output_dir or video.with_name(f'{video.stem}_frames')
    output_dir = output_dir.expanduser().resolve()
    capture = cv2.VideoCapture(str(video))
    frame_index = 0
    saved = 0
    try:
        if not capture.isOpened():
            raise OSError(f'Could not open video {video}')
        while capture.grab():
            if frame_index % options.every == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    raise OSError(f'Could not decode frame {frame_index + 1} of {video}')
                if saved == 0:
                    output_dir.mkdir(parents=True)
                path = output_dir / f'{video.stem}_{frame_index + 1:06d}.png'
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f'Could not write {path}')
                saved += 1
            frame_index += 1
        if saved == 0:
            raise OSError(f'No readable frames in {video}')
    except (OSError, cv2.error) as exc:
        parser.exit(1, f'{exc}\n')
    finally:
        capture.release()

    print(f'Exported {saved} PNGs to {output_dir}')
