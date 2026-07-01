import pathlib as _pl
import site as _site

for _sp in _site.getsitepackages():
    _f = _pl.Path(_sp) / "torchvision" / "_meta_registrations.py"
    if _f.exists():
        src = _f.read_text()
        old = "torchvision.extension._has_ops()"
        new = 'getattr(torchvision, "extension", None) is not None and torchvision.extension._has_ops()'
        if old in src and new not in src:
            _f.write_text(src.replace(old, new))
            print("patched _meta_registrations.py")
        else:
            print("_meta_registrations.py already patched or pattern not found")
        break
