# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

caffe_root='/home/qijie/caffe-master/'

import sys
sys.path.insert(0,caffe_root+'python')
import caffe

#显示图片大小为10,图形的插值以最近为原则，颜色是灰色
plt.rcParams['figure.figsize']=(10,10)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

caffe.set_mode_cpu()
model_def=caffe_root+'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights=caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net=caffe.Net(model_def,model_weights,caffe.TEST)

mu=np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu=mu.mean(1).mean(1)
print 'mean-substracted values:',zip('BGR',mu)

transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(50,  # batch size
                          3,  # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
plt.show()
net.blobs['data'].data[...] = transformed_image
### perform classification
output = net.forward()
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
print 'predicted class is:', output_prob.argmax()

labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels=np.loadtxt(labels_file,str,delimiter='\t')
print 'output label:',labels[output_prob.argmax()]

top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
print 'probabilities and labels:'
print zip(output_prob[top_inds], labels[top_inds])
print '\n\n\n'
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
net.forward()  # run once before timing to set up memory

print 'belows are the parameters of each layers:\n'
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

print 'belows are another orderedDict, net params:\n'
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data);
    plt.show();
    plt.axis('off')

#belows are the first layer filter: conv1
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))

#belows are the first output:conv1
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)

#belows are the fifth layer after pooling :pool5
feat = net.blobs['pool5'].data[0]
vis_square(feat)

#belows are the output values and the histogram of the positive values
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()

#belows are final probability output:prob
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.show()
