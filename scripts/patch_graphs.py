p = r"D:\camera-radar-fusion\scripts\make_graphs.py"
s = open(p).read()
s = s.replace('MODELS = ["b0", "b1", "b2", "b2_filtered", "b0_aug", "b2_aug", "b3"]',
              'MODELS = ["b0", "b1", "b2", "b2_filtered", "b0_aug", "b2_aug", "b2_aug2", "b3", "b3_v2"]')
s = s.replace('["b0_aug", "b2_aug", "b3"], "With augmentation"',
              '["b0_aug", "b2_aug", "b2_aug2", "b3", "b3_v2"], "With augmentation"')
s = s.replace('fm = ["b1", "b2", "b2_filtered", "b2_aug", "b3"]',
              'fm = ["b1", "b2", "b2_filtered", "b2_aug", "b2_aug2", "b3", "b3_v2"]')
s = s.replace('ms = ["b0", "b1", "b2", "b0_aug", "b2_aug", "b3"]',
              'ms = ["b0", "b1", "b2", "b0_aug", "b2_aug", "b2_aug2", "b3", "b3_v2"]')
open(p, "w").write(s)
print("patched")