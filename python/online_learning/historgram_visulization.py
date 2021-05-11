# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

event_file = '/Users/zhouheng/Dropbox/Codes/pimtorch/python/online_learning/runs/Apr29_10-29-45_zhouhengdeMacBook-Pro.local/events.out.tfevents.1619663388.zhouhengdeMacBook-Pro.local.1350.0'

plt.figure(figsize=(10, 10), dpi=150)
plt.ylim((0, 5))
for e in tf.compat.v1.train.summary_iterator(event_file):
  # Then we loop over each value stored for each event
  for v in e.summary.value:
    # Now if the value is the histogram_eval then
    if v.tag == 'a':
      x = list(v.histo.bucket)
      bin = list(v.histo.bucket_limit)
      plt.hist(x, bins=bin)
      plt.show()