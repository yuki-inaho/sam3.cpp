"""Opt-in trained-model lifecycle integration test (all artifacts stay outside Git).

Checks repeated/reset inference, explicit single-object rejection, arbitrary
source-frame starts, and a synthetic blank-frame absence/reappearance. This
behavior test is separate from the frozen native numerical parity protocol.
"""
import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('frames', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--efficient', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = [str(args.executable.resolve()), '--model', str(args.model.resolve()),
               '--cpu', '--n-threads', '4']
    if args.efficient:
        command += ['--experimental-efficient-video']
    results = []
    with (args.output / 'server.log').open('w') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=log, text=True)
        try:
            def read():
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError('Server exited unexpectedly; inspect server.log')
                return json.loads(line)

            def call(text, expected_ok=True):
                process.stdin.write(text + '\n')
                process.stdin.flush()
                response = read()
                if response['ok'] != expected_ok:
                    raise AssertionError((text, response))
                results.append({'command': text, 'result': response})
                return response

            def seed(frame, object_id, source_start=0, sequence_length=17):
                call('load ' + str(frame.resolve()))
                if args.efficient:
                    call(f'track_config {sequence_length} {source_start}')
                call('box 73 245 309 783')
                call('multimask 0')
                return call('track_add ' + str(object_id))

            ready = read()
            assert ready['video_tracking'], ready
            results.append({'ready': ready})
            frame0 = args.frames / '00000.jpg'
            frame1 = args.frames / '00001.jpg'
            initial = seed(frame0, 71)
            call('box 73 245 309 783')
            call('track_add 72', expected_ok=False)
            if args.efficient:
                call('track_config 17 0', expected_ok=False)
            next_frame = call('track_frame ' + str(frame1.resolve()))
            assert next_frame['object_id'] == 71
            call('track_reset')
            call('segment', expected_ok=False)
            call('track_frame ' + str(frame1.resolve()), expected_ok=False)
            repeated_initial = seed(frame0, 71)
            repeated_next = call('track_frame ' + str(frame1.resolve()))
            assert initial == repeated_initial, 'reset changed conditioning output'
            assert next_frame == repeated_next, 'reset changed propagation output'
            if args.efficient:
                call('track_reset')
                arbitrary_start = seed(args.frames / '00008.jpg', 72, source_start=8)
                arbitrary_next = call('track_frame ' + str((args.frames / '00009.jpg').resolve()))
                assert arbitrary_start['frame_index'] == 0
                assert arbitrary_next['frame_index'] == 1 and arbitrary_next['object_id'] == 72
                call('track_reset')
                seed(frame0, 74, sequence_length=1)
                call('track_frame ' + str(frame1.resolve()), expected_ok=False)
                call('track_reset')
                call('track_config 8 8', expected_ok=False)
                seed(frame0, 73)
                black = args.output / 'black.png'
                Image.new('RGB', (480, 848)).save(black)
                absent = call('track_frame ' + str(black.resolve()))
                reentry = call('track_frame ' + str(frame1.resolve()))
                assert absent['object_id'] == reentry['object_id'] == 73
                absent_detection = absent['detections'][0]
                assert absent_detection['object_logit'] < 0
                assert sum(absent_detection['counts'][1::2]) == 0
                assert absent_detection['absent'] and absent_detection['box'] == [0, 0, 0, 0]
                assert reentry['detections'][0]['object_logit'] > 0
                assert sum(reentry['detections'][0]['counts'][1::2]) > 0
                assert not reentry['detections'][0]['absent']
                call('track_reset')
                blank_seed = seed(black, 75)
                blank_detection = blank_seed['detections'][0]
                if blank_detection['object_logit'] <= 0:
                    assert blank_detection['box'] == [0, 0, 0, 0]
                    assert blank_detection['absent'] and sum(blank_detection['counts'][1::2]) == 0
                (args.output / 'blank-seed.json').write_text(json.dumps(blank_seed, indent=2) + '\n')
                (args.output / 'absence-summary.json').write_text(json.dumps(
                    {'absent': absent, 'reentry': reentry}, indent=2) + '\n')
            call('quit')
            process.wait(timeout=10)
            assert process.returncode == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()
    (args.output / 'lifecycle.json').write_text(json.dumps(
        {'passed': True, 'results': results}, indent=2) + '\n')
    print('PASS', args.model, len(results))


if __name__ == '__main__':
    main()
