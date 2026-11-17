p = r"D:\camera-radar-fusion\scripts\make_model_pack.py"
s = open(p).read()
old_f = 'FOLDERS = {"B0": "b0", "B1": "b1", "B2": "b2"}'
old_t = 'TITLES = {"B0": "B0: camera-only YOLO", "B1": "B1: camera + radar (early fusion)", "B2": "B2: camera + radar (mid fusion)"}'
assert old_f in s and old_t in s, "expected lines not found: tell Claude"
s = s.replace(old_f, 'FOLDERS = {"B0": "b0", "B1": "b1", "B2": "b2"} if len(sys.argv) < 3 else {sys.argv[1]: sys.argv[2]}')
s = s.replace(old_t, 'TITLES = {"B0": "B0: camera-only YOLO", "B1": "B1: camera + radar (early fusion)", "B2": "B2: camera + radar (mid fusion)", "B3": "B3: camera + radar + reliability gate (ours)"}')
open(p, "w").write(s)
print("patched")