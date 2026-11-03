"""Writes the runtime's environment (Python, PyTorch, CUDA and cuDNN versions, NumPy, the GPU and its memory) to a JSON file, and stops when no CUDA device is visible."""
import json
import os
import sys
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

def main(argv):
    out, variant = (argv[0], argv[1])
    import torch
    if not torch.cuda.is_available():
        print('torch sees no CUDA device: this notebook needs a GPU runtime')
        return 2
    from train import emul_core as ec
    rep = ec.env_report('cuda')
    rep['torch_variant'] = variant
    rep['runtime_tag'] = os.environ.get('COLAB_RELEASE_TAG') or 'not shown by the VM'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + '.partial', 'w', encoding='ascii', newline='\n') as f:
        json.dump(rep, f, indent=1, sort_keys=True, default=str)
        f.write('\n')
    os.replace(out + '.partial', out)
    g = rep.get('gpu', {})
    print('FIVE VERSIONS: torch %s, CUDA %s, cuDNN %s, Python %s, numpy %s' % (rep.get('torch'), rep.get('torch_cuda'), rep.get('cudnn'), rep.get('python'), rep.get('numpy')))
    print('GPU %s, capability %s, memory %.2f GB; runtime tag %s' % (g.get('name'), g.get('capability'), float(g.get('total_memory_bytes') or 0) / 1000000000.0, rep['runtime_tag']))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
