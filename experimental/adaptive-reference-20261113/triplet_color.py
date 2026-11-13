"""Explicit RGB8 training preparation; separate from the product image loader."""
import io
from PIL import ImageCms


def to_srgb(image):
    if image.mode!='RGB':raise ValueError('Academic pilot expects RGB inputs')
    if image.getexif().get(274,1) not in (None,1):raise ValueError('Unexpected image orientation')
    bits=getattr(image,'tag_v2',{}).get(258,())
    bits=[bits] if isinstance(bits,int) else list(bits)
    source=image.copy();profile_bytes=image.info.get('icc_profile')
    info=dict(original_mode=image.mode,tiff_bits_per_sample=bits,output_precision='PIL RGB8; high-bit TIFF decoded/quantized before ICC processing')
    if profile_bytes:
        try:
            profile=ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes));srgb=ImageCms.createProfile('sRGB')
            result=ImageCms.profileToProfile(source,profile,srgb,renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,outputMode='RGB')
            info.update(policy='embedded ICC to sRGB',source_profile=(ImageCms.getProfileName(profile)+' '+ImageCms.getProfileDescription(profile)).strip())
        except Exception as exc:raise ValueError('Cannot honor embedded ICC profile') from exc
    else:
        result=source;info.update(policy='untagged assumed sRGB',source_profile=None)
    result.info['icc_profile']=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    return result,info
