import json
import struct
import zlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from reference_color_app.media import decode_image, decode_frame, probe_media, ffmpeg_path
from reference_color_app.pipeline import analyze, json_safe


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def image(self, name='image.png'):
        rng = np.random.default_rng(42)
        p = self.root/name
        Image.fromarray(rng.integers(0,256,(77,113,3),dtype=np.uint8)).save(p)
        return p

    def video(self, hdr=False):
        p = self.root/('hdr.mp4' if hdr else 'clip.mp4')
        args = [str(ffmpeg_path()),'-hide_banner','-nostdin','-y','-f','lavfi','-i',
                'color=red:size=96x64:rate=4:duration=1','-f','lavfi','-i',
                'color=blue:size=96x64:rate=4:duration=1','-filter_complex',
                '[0:v][1:v]concat=n=2:v=1:a=0[v]','-map','[v]','-an','-c:v','libx264']
        if hdr:
            args += ['-color_trc','smpte2084','-color_primaries','bt2020','-colorspace','bt2020nc',
                     '-x264-params','colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc']
        subprocess.run(args+[str(p)],check=True,capture_output=True,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return p

    def test_full_resolution_and_exif(self):
        p=self.root/'oriented.jpg'
        exif=Image.Exif();exif[274]=6
        Image.new('RGB',(1501,83),'red').save(p,exif=exif)
        a=decode_image(p)
        self.assertEqual(a.shape,(1501,83,3))
        self.assertEqual(a.dtype,np.uint8)

    def test_video_times_no_audio_and_hdr(self):
        p=self.video()
        meta=probe_media(p)
        self.assertAlmostEqual(meta['duration_seconds'],2)
        self.assertFalse(meta['has_audio'])
        for time,channel in [(0,0),(.75,0),(1.25,2),(1.99,2)]:
            rgb,info=decode_frame(p,time)
            self.assertEqual(rgb.shape,(64,96,3))
            self.assertGreater(rgb[...,channel].mean(),240)
            self.assertEqual(info['requested_time_seconds'],time)
            self.assertIsInstance(info['actual_frame_pts_seconds'],float)
            self.assertAlmostEqual(info['actual_frame_pts_seconds'],1.75 if time==1.99 else time)
        hdr=self.video(True)
        self.assertIn('smpte2084',probe_media(hdr)['video_stream'])
        with self.assertRaisesRegex(ValueError,'HDR'):
            decode_frame(hdr,0)

    def test_bad_files_and_time(self):
        p=self.root/'bad';p.write_text('not media')
        for fn in [decode_image,probe_media]:
            with self.assertRaises(ValueError):fn(p)
        p=self.video()
        for t in [-1,2,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):decode_frame(p,t)
        with self.assertRaisesRegex(ValueError,'reference'):
            analyze(self.image(),p)

    def test_high_precision_image_is_not_silently_clipped(self):
        for name,array in [('sixteen.tif',np.array([[0,512,65535]],np.uint16)),
                           ('float.tif',np.array([[0.,.5,1.]],np.float32))]:
            p=self.root/name;Image.fromarray(array).save(p)
            for fn in (probe_media,decode_image):
                with self.assertRaisesRegex(ValueError,'8-bit SDR'):fn(p)

    def test_encoded_16bit_rgb_rgba_png(self):
        def chunk(tag,data):
            return struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data)&0xffffffff)
        for channels,color_type in ((3,2),(4,6)):
            values=[512,32768,65535]+([65535] if channels==4 else [])
            p=self.root/f'rgb{channels}-16.png'
            raw=b'\x00'+struct.pack('>'+'H'*channels,*values)
            p.write_bytes(b'\x89PNG\r\n\x1a\n'+
                chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,16,color_type,0,0,0))+
                chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
            with Image.open(p) as im:
                self.assertEqual(im.mode,'RGB' if channels==3 else 'RGBA')
                im.load()  # Fixture is a valid decoded image, not malformed data.
            for fn in (probe_media,decode_image):
                with self.assertRaisesRegex(ValueError,'8-bit SDR'):fn(p)

    def test_encoded_16bit_rgb_tiff_and_8bit_controls(self):
        # Minimal valid uncompressed TIFF with three 16-bit samples per pixel.
        entries=[(256,4,1,1),(257,4,1,1),(258,3,3,134),(259,3,1,1),
                 (262,3,1,2),(273,4,1,140),(277,3,1,3),(278,4,1,1),
                 (279,4,1,6),(284,3,1,1)]
        p=self.root/'rgb16.tif'
        p.write_bytes(b'II'+struct.pack('<HIH',42,8,len(entries))+
                      b''.join(struct.pack('<HHII',*entry) for entry in entries)+
                      struct.pack('<IHHHHHH',0,16,16,16,512,32768,65535))
        with Image.open(p) as im:
            self.assertEqual(im.mode,'RGB');self.assertEqual(im.tag_v2[258],(16,16,16));im.load()
        for fn in (probe_media,decode_image):
            with self.assertRaisesRegex(ValueError,'8-bit SDR'):fn(p)
        for mode in ('RGB','RGBA','L','P','CMYK'):
            p=self.root/f'control-{mode}.tif'
            Image.new(mode,(13,7)).save(p)
            self.assertEqual(probe_media(p)['kind'],'image')
            self.assertEqual(decode_image(p).shape,(7,13,3))

    def test_encoded_video_component_precision(self):
        # Real encoded SDR, independent of HDR tags. FFV1 preserves depth.
        for fmt in ('yuv444p10le','yuv444p12le','gbrp16le','gray16le','yuv420p','bgr0','gray'):
            with self.subTest(fmt=fmt):
                p=self.root/(fmt+'.mkv')
                subprocess.run([str(ffmpeg_path()),'-hide_banner','-nostdin','-y',
                    '-f','lavfi','-i','color=red:size=16x16:rate=4:duration=1',
                    '-c:v','ffv1','-pix_fmt',fmt,'-colorspace','bt709',str(p)],
                    check=True,capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                meta=probe_media(p)
                self.assertFalse(meta['hdr_unsupported'])
                if fmt in ('yuv444p12le','gbrp16le','gray16le'):
                    with self.assertRaisesRegex(ValueError,'暂不支持此视频位深'):decode_frame(p,0)
                else:
                    rgb,decoded=decode_frame(p,0)
                    self.assertEqual(rgb.dtype,np.uint8)
                    self.assertEqual(decoded['working_precision_bits'],8)
                    self.assertEqual(decoded['precision_conversion'],fmt=='yuv444p10le')
                    if fmt=='yuv444p10le':
                        self.assertEqual(decoded['component_bit_depths'],[10,10,10])
                        self.assertIn('10位 SDR',decoded['precision_note'])
                        self.assertGreater(float(rgb[...,0].mean()),240)

    def test_descriptor_precision_and_unknown_fail_closed(self):
        from reference_color_app.media import _video_precision
        for fmt,expected in [('rgb24',[8,8,8]),('rgba',[8,8,8,8]),
                             ('p010le',[10,10,10]),('rgb48le',[16,16,16]),
                             ('gbrpf32le',[32,32,32]),('yuv444p12le',[12,12,12]),
                             ('unrecognized_precision',None)]:
            with self.subTest(fmt=fmt):
                name,depths=_video_precision('Stream #0:0: Video: rawvideo, '+fmt+', 16x16, 4 fps')
                self.assertEqual(name,fmt);self.assertEqual(depths,expected)
        p=self.video()
        meta=probe_media(p);meta['component_bit_depths']=None
        with patch('reference_color_app.media.probe_media',return_value=meta):
            with self.assertRaisesRegex(ValueError,'无法确认视频位深'):decode_frame(p,0)

    def test_actual_chain_repeat_diagnostics(self):
        p=self.image(); ref=self.image('reference.png')
        first=analyze(p,ref);second=analyze(p,ref)
        self.assertEqual(first.layers.shape,(2,64,64,64,3))
        self.assertTrue(np.isfinite(first.layers).all())
        self.assertEqual(first.diagnostics,second.diagnostics)
        d=first.diagnostics
        self.assertEqual(d['source_packet']['mode'],0)
        self.assertEqual(d['reference_packet']['mode'],1)
        self.assertEqual(len(d['parameter_buffer']),37)
        self.assertEqual(set(d['matrices']) & set('TRAFC'),set('TRAFC'))
        self.assertEqual(len(d['algorithm_fingerprint']),64)
        self.assertEqual(len(d['layer_sha256']),64)
        json.dumps(d,allow_nan=False)
        self.assertIsNone(json_safe(np.array([np.nan]))[0])

    def test_direction_and_full_input(self):
        p=self.image(); ref=self.image('ref.png')
        from reference_color_app.features import extract_features
        seen=[]
        def capture(rgb,mode):
            seen.append((rgb.shape,mode));return extract_features(rgb,mode)
        with patch('reference_color_app.pipeline.extract_features',side_effect=capture):
            analyze(p,ref)
        self.assertEqual(seen,[((77,113,3),0),((77,113,3),1)])

    def test_video_chain_actual_frame_and_repeat(self):
        p=self.video();ref=self.image()
        a=analyze(p,ref,1.25);b=analyze(p,ref,1.25)
        from reference_color_app.features import extract_features
        frame,metadata=decode_frame(p,1.25)
        self.assertEqual(a.diagnostics['source_packet']['values'],
                         extract_features(frame,0).values.tolist())
        self.assertEqual(a.diagnostics['inputs']['target']['metadata'],metadata)
        self.assertEqual(a.diagnostics,b.diagnostics)


if __name__=='__main__':unittest.main()


class ProbeDimensionRegression(unittest.TestCase):
    def test_fourcc_is_not_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.bin'; path.write_bytes(b'not an image')
            log='Duration: 00:00:02.00\nStream #0:0: Video: ffv1 (FFV1 / 0x31564646), bgr0, 13x9, 4 fps\n'
            with patch('reference_color_app.media._run',return_value=(1,b'',log)):
                metadata=probe_media(path)
            self.assertEqual((metadata['width'],metadata['height']),(13,9))
            invalid='Duration: 00:00:02.00\nStream #0:0: Video: h264 (avc1 / 0x1), yuv420p, 0x9, 4 fps\n'
            with patch('reference_color_app.media._run',return_value=(1,b'',invalid)):
                with self.assertRaises(ValueError): probe_media(path)
