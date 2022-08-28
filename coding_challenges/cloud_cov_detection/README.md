# Cloud coverage detection

This is a task for cloud coverage detection on a satellite image. The algorithm should be relatively fast and should decide whether to accept or reject an image based on a predicted cloud coverage ratio threshold.

I've taken a Unet implementation along with the Landsatsatellite image dataset and added several improvements/corrections to the pipeline, along with a confusion matrix, pr-curve and F1 score evaluation. 

## Training
You can train several epochs on gpu in 10 minutes, which gives a decent baseline performance. Sometimes training fails and gives bad results, so just restart it.

## Quantization
I've also used post-training static quantization to quantize the UNET to an int8 model which takes up 4 times less memory and is roughly 1.5-2x faster with almost no performance drop. If performance drop is an issue, one can use quantized-aware static training which requires just adding a few more simple steps, and training the model.

# Performance
The overall performance is good, but there are failure cases on snowy mountains, shown at the very end of the notebook. The dataset also has issues because you get fully cloud covered images whose ground truth says that there are no clouds. It is also difficult to label cloud coverage due to ambiguity so in this regard, the labels are noisy.

## Notes
If you want to run the notebook I recommend at least 24GB of ram and a GPU with 6GB (or just reduce the batchsize). I've added cumulative batch training, which means that you can use smaller batch sizes with accumulating gradients which emulate larger batch sizes. This isn't entirely true though because the batch normalization statistics are still calculated on the true minibatches, but this isn't much of an issue.


