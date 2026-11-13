"""Train-only deterministic references; filenames are not used as style labels."""
import hashlib


def choose_training_reference(target,training_contents,index):
    paths=sorted(set(training_contents))
    if target not in paths or len(paths)<2 or type(index)!=int or index<0:
        raise ValueError('Need target and another content in the training-only pool')
    choices=[path for path in paths if path!=target]
    key=hashlib.sha256(('yingse-consistent-reference-v1:'+str(index)).encode()).digest()
    return choices[int.from_bytes(key[:8],'big')%len(choices)]


def use_synthetic_reference(style_index,epoch):
    if type(style_index)!=int or not 0<=style_index<4 or type(epoch)!=int or not 1<=epoch<=8:
        raise ValueError('Expected four training styles and epoch1..8')
    return (style_index+epoch)%2==0
