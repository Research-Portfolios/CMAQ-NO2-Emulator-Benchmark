"""The font choice of the figure style: finds the installed font families and resolves the sans-serif family that the figures use."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SANS_PREFERENCE = ['Arial', 'Helvetica', 'Nimbus Sans', 'Liberation Sans', 'DejaVu Sans']
RESOLVED_FAMILY = None
FONT_DIR = ROOT / 'fonts'
FONT_ENV = 'FIGURE_FONT'

def font_file_families(font_dir=None):
    from matplotlib import font_manager
    font_dir = FONT_DIR if font_dir is None else Path(font_dir)
    families = {}
    if not font_dir.is_dir():
        return families
    for path in sorted(font_dir.rglob('*')):
        if path.suffix.lower() not in ('.ttf', '.otf'):
            continue
        font_manager.fontManager.addfont(str(path))
        name = font_manager.FontProperties(fname=str(path)).get_name()
        families.setdefault(name, []).append(path)
    return families

def resolve_family(preference=None):
    from matplotlib import font_manager
    requested = os.environ.get(FONT_ENV, '').strip()
    if requested and preference is None:
        from_files = font_file_families()
        if requested not in from_files:
            raise RuntimeError(FONT_ENV + "='" + requested + "' is not a family provided by a font file under fonts/ (found: " + (', '.join(sorted(from_files)) or 'none') + '). Refusing to substitute. Put the font file under fonts/ or unset the variable.')
        return requested
    preference = preference or SANS_PREFERENCE
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preference:
        if name in installed:
            return name
    raise RuntimeError('None of the requested figure fonts is installed on this machine: ' + ', '.join(preference) + '. Refusing to let matplotlib substitute silently, because the substitute would differ between this machine and Colab and nothing downstream would show it. Install one, or amend SANS_PREFERENCE in report/figure_style.py.')
