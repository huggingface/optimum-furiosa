# Optimum Furiosa documentation

1. Setup
```bash
pip install hf-doc-builder==0.4.0 watchdog --upgrade
```

2. Local Development
```bash
doc-builder preview optimum.furiosa docs/source/
```
3. Build Docs
```bash
doc-builder build optimum.furiosa docs/source/ --build_dir build/ 
```

## Add assets/Images

Adding images/assets is only possible through `https://` links meaning you need to use `https://raw.githubusercontent.com/huggingface/optimum-furiosa/main/docs/assets/` prefix.
