import re

pos = ["happy","good","excellent","positive","love"]
neg = ["sad","bad","terrible","negative"]

def h(t, kws):
    for k in kws:
        if re.search(rf"\b{k}\b", t, flags=re.IGNORECASE):
            return True
    return False

for line in open("input.txt", "r", encoding="utf-8"):
    s = line.strip()
    if not s:
        continue
    p = h(s, pos)
    n = h(s, neg)
    if p and not n:
        print("positive")
    elif n and not p:
        print("negative")
    else:
        print("neutral")
