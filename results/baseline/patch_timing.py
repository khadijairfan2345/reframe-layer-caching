p = "inference/inference.py"
s = open(p).read()

def rep(old, new):
    global s
    assert s.count(old) == 1, "not found exactly once: " + old
    s = s.replace(old, new)

rep("    timer = []\n", "    timer = []\n    frame_ids = []\n")
rep("            start_time = time.time()\n",
    "            torch.cuda.synchronize()\n            start_time = time.time()\n")
rep("            end_time = time.time()\n            timer.append(end_time - start_time)\n",
    "            torch.cuda.synchronize()\n            end_time = time.time()\n            timer.append(end_time - start_time)\n            frame_ids.append(frame_idx)\n")
rep("        # Ignore the first one\n        timer = timer[1:]\n",
    "        import csv\n        with open(str(pred_output_dir / 'frame_times.csv'), 'w', newline='') as f:\n            w = csv.writer(f)\n            w.writerow(['frame', 'seconds'])\n            w.writerows(zip(frame_ids, timer))\n        # Ignore the first one\n        timer = timer[1:]\n")
open(p, "w").write(s)
print("patched")
