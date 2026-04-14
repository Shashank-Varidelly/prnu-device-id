import json
from pathlib import Path

DATA = Path("/content/prnu-device-id/data/raw")
OUT  = Path("/content/prnu-device-id/data/splits_demo.json")

DEVICES = [
    "Canon_Ixus55_0", "Canon_Ixus70_0", "Nikon_D200", "Nikon_CoolPixS710_0",
    "Sony_DSC_H50", "Sony_DSC_T77", "Sony_DSC_W170", "Samsung_NV15"
]

splits = {"dresden": {"train":{}, "val":{}, "test":{}, "fingerprint":{},
                      "test_wa":{}, "test_fb":{}}}

for split in splits["dresden"]:
    for dev in DEVICES:
        folder = DATA/split/dev
        splits["dresden"][split][dev] = [str(p) for p in sorted(folder.glob("*.JPG"))]

OUT.write_text(json.dumps(splits, indent=2))
print("Wrote", OUT)
for k,v in splits["dresden"].items():
    print(f"  {k:<15}: {sum(len(x) for x in v.values())} images")