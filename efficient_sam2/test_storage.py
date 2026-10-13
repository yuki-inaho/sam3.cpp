"""Independent GGUF fixture writer and native storage/rotation contract tests.

Only Python's standard library is needed. Fixtures are generated in a temporary
folder, contain no trained/private weights, and never enter the repository.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())
DTYPES = {'F32': (0, 4), 'F16': (1, 2), 'I8': (24, 1), 'U8': (24, 1), 'BF16': (30, 2)}

def string(value):
    encoded = value.encode()
    return struct.pack('<Q', len(encoded)) + encoded

def encode_json(value):
    return json.dumps(value, separators=(',', ':'), sort_keys=True)

def write_fixture(path, records, changes=None, map_change=None, layer_change=None):
    mapping = {}
    payload = bytearray()
    directory = bytearray()
    for name, dtype, shape, raw in records:
        physical = 't_' + hashlib.sha256(name.encode()).hexdigest()[:60]
        mapping[physical] = {'name': name, 'dtype': dtype, 'shape': shape}
        dims = list(reversed(shape)) if 1 <= len(shape) <= 4 else [math.prod(shape)]
        payload.extend(b'\0' * (-len(payload) % 32))
        directory.extend(string(physical) + struct.pack('<I', len(dims)))
        directory.extend(struct.pack('<' + 'Q' * len(dims), *dims))
        directory.extend(struct.pack('<IQ', DTYPES[dtype][0], len(payload)))
        payload.extend(raw)
    if map_change:
        map_change(mapping)
    layers = {}
    tensors = {name: (dtype, shape, raw) for name, dtype, shape, raw in records}
    for name, dtype, shape, _ in records:
        if dtype == 'I8' and (name == 'weight' or name.endswith('.weight')):
            prefix = name[:-7] if name != 'weight' else ''
            key = lambda suffix: prefix + '.' + suffix if prefix else suffix
            try:
                marker = json.loads(tensors[key('comfy_quant')][2])
                group = marker.get('convrot_groupsize', 1) if marker.get('convrot') else 1
            except (ValueError, KeyError, AttributeError):
                group = 4
            layers[prefix] = {'in_features': shape[1], 'out_features': shape[0],
                'group_size': group, 'has_bias': key('bias') in tensors,
                'source_dtype': 'float32', 'activation_mode': 'w8a16'}
    if layer_change:
        layer_change(layers)
    metadata = {
        'general.architecture': (8, 'efficientsam2'),
        'general.alignment': (4, 32),
        'efficient_sam2.storage_version': (4, 1),
        'efficient_sam2.convrot_basis': (8, 'regular-h4-kron'),
        'efficient_sam2.test_fixture': (7, True),
        'efficient_sam2.model_config': (8, encode_json({'variant': 'base_plus', 'image_size': 1024, 'sam2_version': '2.1'})),
        'efficient_sam2.tensor_map': (8, encode_json(mapping)),
        'efficient_sam2.quantization': (8, encode_json({'scheme': 'symmetric-per-output-channel-int8',
            'layers': layers})),
        'efficient_sam2.int8_layers': (4, len(layers)),
        'efficient_sam2.convrot_layers': (4, sum(isinstance(x.get('group_size'), (float, int)) and x['group_size'] > 1 for x in layers.values())),
    }
    metadata.update(changes or {})
    metadata = {key: value for key, value in metadata.items() if value is not None}
    header = bytearray(struct.pack('<4sIQQ', b'GGUF', 3, len(records), len(metadata)))
    for name, (kind, value) in metadata.items():
        header.extend(string(name) + struct.pack('<I', kind))
        header.extend(string(value) if kind == 8 else struct.pack('<?' if kind == 7 else '<I', value))
    header.extend(directory)
    header.extend(b'\0' * (-len(header) % 32))
    path.write_bytes(header + payload)

def records(group=4, scale=0.25, marker_extra=None):
    marker = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': group, 'per_row': True}
    marker.update(marker_extra or {})
    descriptor = encode_json(marker).encode()
    return [
        ('layer.weight', 'I8', [2, 4], struct.pack('<8b', 1, 2, 3, 4, -2, 5, 7, -3)),
        ('layer.weight_scale', 'F32', [2, 1], struct.pack('<2f', scale, 0.5)),
        ('layer.comfy_quant', 'U8', [len(descriptor)], descriptor),
        ('layer.bias', 'F32', [2], struct.pack('<2f', 2, 3)),
    ]

class Contract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'fixture.gguf'
        self.out = Path(self.temp.name) / 'values.f32'
    def tearDown(self):
        self.temp.cleanup()
    def run_native(self, name=None):
        args = [BINARY, str(self.path)]
        if name:
            args += [name, str(self.out)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        self.assertGreaterEqual(result.returncode, 0, result.stderr)
        return result
    def test_regular_h4_inverse(self):
        write_fixture(self.path, records())
        result = self.run_native('layer.weight')
        self.assertEqual(result.returncode, 0, result.stderr)
        got = struct.unpack('<8f', self.out.read_bytes())
        basis = [[1,1,1,-1], [1,1,-1,1], [1,-1,1,1], [-1,1,1,1]]
        expected = [sum(row[k]*basis[k][j]/2 for k in range(4))*scale
                    for row, scale in [([1,2,3,4], .25), ([-2,5,7,-3], .5)] for j in range(4)]
        for a, b in zip(got, expected): self.assertAlmostEqual(a,b,places=6)
    def test_preserves_f32_bias(self):
        write_fixture(self.path, records())
        self.assertEqual(self.run_native('layer.bias').returncode, 0)
        self.assertEqual(struct.unpack('<2f', self.out.read_bytes()), (2,3))
    def test_rejects_unknown_architecture(self):
        write_fixture(self.path, records(), {'general.architecture': (8, 'sam3_1_multiplex')})
        self.assertNotEqual(self.run_native().returncode, 0)
    def test_native_profile_requires_exact_declared_policy(self):
        profile = {'variant': 'base_plus', 'image_size': 1024, 'sam2_version': '2.1',
                   'efficient_swr': True, 'efficient_smr': True, 'bypass_type': 'bottleneck',
                   'selected_layers': list(range(6, 21)), 'final_global_layer': 20,
                   'memory_temporal_stride': 1, 'memory_drop_ratio': .95, 'wb_theta': .7,
                   'dilate_mask': True, 'dilate_kernel_size': 5}
        def inspect(config):
            write_fixture(self.path, records(), {'efficient_sam2.model_config': (8, encode_json(config))})
            return subprocess.run([BINARY, str(self.path), '--native-profile'], capture_output=True, text=True)
        self.assertEqual(inspect(profile).returncode, 0)
        for field, bad in [('efficient_swr', False), ('efficient_smr', False),
                           ('memory_temporal_stride', 2), ('memory_drop_ratio', .90),
                           ('wb_theta', .8), ('dilate_mask', False), ('dilate_kernel_size', 3),
                           ('selected_layers', list(range(7, 21))), ('bypass_type', 'linear'),
                           ('image_size', 512), ('final_global_layer', 16)]:
            with self.subTest(field=field):
                self.assertNotEqual(inspect(dict(profile, **{field: bad})).returncode, 0)
        without_policy = {k: profile[k] for k in ('variant', 'image_size', 'sam2_version')}
        self.assertNotEqual(inspect(without_policy).returncode, 0)
    def test_rejects_unknown_quantization_semantics(self):
        write_fixture(self.path, records(marker_extra={'mystery': True}))
        self.assertNotEqual(self.run_native().returncode, 0)
    def test_rejects_zero_scale(self):
        write_fixture(self.path, records(scale=0))
        self.assertNotEqual(self.run_native().returncode, 0)
    def test_rejects_nonfinite_scale(self):
        write_fixture(self.path, records(scale=float('nan')))
        self.assertNotEqual(self.run_native().returncode, 0)
    def test_rejects_missing_scale(self):
        write_fixture(self.path, [x for x in records() if not x[0].endswith('weight_scale')])
        self.assertNotEqual(self.run_native().returncode, 0)
    def test_rejects_wrong_group(self):
        write_fixture(self.path, records(group=8))
        self.assertNotEqual(self.run_native().returncode, 0)
    def test_rejects_same_element_count_wrong_shape(self):
        def change(mapping):
            for value in mapping.values():
                if value['name']=='layer.weight': value['shape']=[4,2]
        write_fixture(self.path, records(), map_change=change)
        self.assertNotEqual(self.run_native().returncode, 0)
    def test_rejects_truncated_payload(self):
        write_fixture(self.path, records())
        self.path.write_bytes(self.path.read_bytes()[:-1])
        self.assertNotEqual(self.run_native().returncode, 0)
    def test_rejects_wrong_basis(self):
        write_fixture(self.path, records(), {'efficient_sam2.convrot_basis': (8, 'sylvester')})
        self.assertNotEqual(self.run_native().returncode, 0)
    def test_rejects_w8a8_activation_policy(self):
        write_fixture(self.path, records(), layer_change=lambda layers: layers['layer'].update(activation_mode='w8a8'))
        result = self.run_native()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not W8A8', result.stderr)
    def test_unrotated_group_identity(self):
        rs=records()
        marker=encode_json({'format':'int8_tensorwise','convrot':False,'per_row':True}).encode()
        rs[2]=('layer.comfy_quant','U8',[len(marker)],marker)
        write_fixture(self.path,rs)
        result=self.run_native('layer.weight')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(struct.unpack('<8f',self.out.read_bytes()),(.25,.5,.75,1,-1,2.5,3.5,-1.5))
    def test_restores_bfloat16(self):
        write_fixture(self.path, [('norm.weight','BF16',[2],struct.pack('<2H',0x3f80,0xc000))])
        result=self.run_native('norm.weight')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(struct.unpack('<2f',self.out.read_bytes()),(1,-2))
    def test_rejects_quantized_conv2d(self):
        rs=records()
        rs[0]=('layer.weight','I8',[1,4,1,2],struct.pack('<8b',1,10,2,20,3,30,4,40))
        rs[1]=('layer.weight_scale','F32',[1,1,1,1],struct.pack('<f',1))
        write_fixture(self.path,rs)
        self.assertNotEqual(self.run_native().returncode,0)

    def test_rejects_layer_metadata(self):
        for field, values in {
            'in_features': [0, 8, -1, True, '4', 1.5],
            'out_features': [0, 4, False],
            'group_size': [0, 1, 8, True, '4'],
            'has_bias': [False, 1, 'true'],
            'source_dtype': ['int8', '', None, 32],
            'activation_mode': ['unknown', None],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    write_fixture(self.path, records(), layer_change=lambda layers: layers['layer'].update({field: value}))
                    self.assertNotEqual(self.run_native().returncode,0)
        for field in ('in_features','out_features','group_size','has_bias','source_dtype','activation_mode'):
            with self.subTest(missing=field):
                write_fixture(self.path, records(), layer_change=lambda layers: layers['layer'].pop(field))
                self.assertNotEqual(self.run_native().returncode,0)

    def test_rejects_count_metadata(self):
        for key in ('int8_layers','convrot_layers'):
            for entry in (None,(4,0),(4,2),(8,'1'),(7,True)):
                with self.subTest(key=key, entry=entry):
                    write_fixture(self.path, records(), {'efficient_sam2.'+key: entry})
                    self.assertNotEqual(self.run_native().returncode,0)

    def test_rejects_bias_mismatch(self):
        for dtype, shape, data in [('F32',[1,2],struct.pack('<2f',2,3)),
                                    ('F16',[2],struct.pack('<2e',2,3)),
                                    ('F32',[2],struct.pack('<2f',float('inf'),3))]:
            with self.subTest(dtype=dtype, shape=shape):
                rs=records(); rs[-1]=('layer.bias',dtype,shape,data)
                write_fixture(self.path,rs)
                self.assertNotEqual(self.run_native().returncode,0)
        write_fixture(self.path, records()[:-1], layer_change=lambda layers: layers['layer'].update(has_bias=True))
        self.assertNotEqual(self.run_native().returncode,0)

    def test_rejects_scale_schema(self):
        for dtype, shape, data in [('F32',[2],struct.pack('<2f',1,2)),
                                    ('F32',[1],struct.pack('<f',1)),
                                    ('F16',[2,1],struct.pack('<2e',1,2))]:
            rs=records(); rs[1]=('layer.weight_scale',dtype,shape,data)
            write_fixture(self.path,rs)
            self.assertNotEqual(self.run_native().returncode,0)

    def test_rejects_malformed_descriptors(self):
        for marker in [b'{}',b'[]',b'null',b'{',b'\xff',b'"string"',
            b'{"format":"int8_tensorwise","format":"int8_tensorwise"}',
            b'{"format":"int8_tensorwise","convrot":false,"per_row":true,"per_row":true}']:
            with self.subTest(marker=marker):
                rs=records(); rs[2]=('layer.comfy_quant','U8',[len(marker)],marker)
                write_fixture(self.path,rs)
                self.assertNotEqual(self.run_native().returncode,0)
        for field,value in [('convrot',1),('per_row',False),('convrot_groupsize',True)]:
            write_fixture(self.path,records(marker_extra={field:value}))
            self.assertNotEqual(self.run_native().returncode,0)
        for dtype, shape in [('I8',None),('U8','matrix')]:
            rs=records(); name,_,dims,data=rs[2]
            rs[2]=(name,dtype,dims if shape is None else [1,len(data)],data)
            write_fixture(self.path,rs)
            self.assertNotEqual(self.run_native().returncode,0)

    def test_rejects_duplicate_and_invalid_json_metadata(self):
        for key in ('model_config','quantization','tensor_map'):
            for data in ('{"x":1,"x":2}', '[]', 'null', '{', '{"x":1e999}'):
                with self.subTest(key=key,data=data):
                    write_fixture(self.path,records(),{'efficient_sam2.'+key:(8,data)})
                    self.assertNotEqual(self.run_native().returncode,0)

    def test_rejects_malformed_gguf_without_crash(self):
        for changes in ({'general.alignment':(8,'32')}, {'general.alignment':(4,0)},
                        {'general.alignment':(4,64)}, {'':(4,1)}):
            write_fixture(self.path,records(),changes)
            self.assertNotEqual(self.run_native().returncode,0)
        for shape in ([2,0],[0,4]):
            rs=records(); rs[0]=('layer.weight','I8',shape,b'')
            write_fixture(self.path,rs)
            self.assertNotEqual(self.run_native().returncode,0)
        write_fixture(self.path,records())
        original=self.path.read_bytes()
        for end in (0,4,23,24,32,64,len(original)//2,len(original)-1):
            with self.subTest(truncate=end):
                self.path.write_bytes(original[:end])
                self.assertNotEqual(self.run_native().returncode,0)
        physical='t_'+hashlib.sha256(b'layer.weight').hexdigest()[:60]
        directory_at=original.index(string(physical))
        # Enormous string lengths, counts, rank, zero dimension, and offset.
        for at, fmt, value in [(8,'Q',2**64-1),(16,'Q',2**64-1),(24,'Q',2**64-1),
            (directory_at+len(string(physical)),'I',5),
            (directory_at+len(string(physical))+4,'Q',2**63),
            (directory_at+len(string(physical))+24,'Q',2**64-1)]:
            with self.subTest(at=at,value=value):
                corrupted=bytearray(original); struct.pack_into('<'+fmt,corrupted,at,value)
                self.path.write_bytes(corrupted)
                self.assertNotEqual(self.run_native().returncode,0)

    def test_root_linear_and_bias_free(self):
        rs=[(name.removeprefix('layer.'),dtype,shape,data) for name,dtype,shape,data in records()[:-1]]
        write_fixture(self.path,rs)
        result=self.run_native('weight')
        self.assertEqual(result.returncode,0,result.stderr)

    def test_float_only_path(self):
        rs=[('conv.weight','F32',[1,1,1,2],struct.pack('<2f',1,-2)),
            ('norm.weight','F16',[2],struct.pack('<2e',3,4))]
        write_fixture(self.path,rs)
        for name,expected in [('conv.weight',(1,-2)),('norm.weight',(3,4))]:
            result=self.run_native(name)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(struct.unpack('<2f',self.out.read_bytes()),expected)

    def test_rejects_orphan_and_unindexed_layers(self):
        write_fixture(self.path,records(),layer_change=lambda layers: layers.update(ghost=dict(layers['layer'])))
        self.assertNotEqual(self.run_native().returncode,0)
        for suffix in ('weight_scale','comfy_quant'):
            rs=records(); item=next(r for r in rs if r[0].endswith(suffix))
            rs.append(('orphan.'+suffix,*item[1:]))
            write_fixture(self.path,rs)
            self.assertNotEqual(self.run_native().returncode,0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
