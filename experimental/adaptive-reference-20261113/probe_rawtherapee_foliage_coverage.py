"""Run the frozen coverage screen on three verified outdoor photos."""
import argparse
from pathlib import Path

from probe_rawtherapee_room_coverage import run

SOURCES = {
    'rawtherapee-foliage-input-v1/_6I_8176.CR3': 'dc7bb82517a992e23a6c89b38b7120ca88ff6a5c09553d6e3a3a8cafd7a92d10',
    'rawtherapee-foliage-input-v1/E5Dii252_2021_12_04_1546.CR2': '64de7d90ebdf8757e2be82e9b958b811c7ccdc37bc4a60a9a1bf83931b24c430',
    'rawtherapee-foliage-input-v1/IMG_9562.CR2': '93aa81d09c1ad583ec4af3c92943ab8f8dd5cb01e80f6bbf764545a3d06d3073',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'cli', 'response_report', 'profile_report', 'expansion_report', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()), sources=SOURCES, output_name='foliage-coverage.png')
