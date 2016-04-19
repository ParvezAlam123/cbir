import cv2
import json
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from sklearn import preprocessing
from skimage.feature import greycomatrix, greycoprops
from cbir.cvclasses.localbinarypatterns import LocalBinaryPatterns
from cbir.cvclasses.searcher import Searcher

from .forms import ImageForm
from .models import Image


# Create your views here.
def home(request):

    title = "Upload Image"
    context = {
        "template_title": title
    }

    return render(request, "home.html", context)


# image uploader / drop-zone handler
def upload_image(request):
    form = ImageForm(request.POST, request.FILES or None)

    if form.is_valid():
        pic = Image(file=request.FILES['file'])

        pic.save()              # save image first so we can retrieve the file path

        data = process_image(pic)      # process the image; extract features

        return HttpResponse(json.dumps(data), content_type="application/json")

    return HttpResponseBadRequest('Image upload form not valid')


# extract features from query image
def process_image(pic):
    img = cv2.imread(pic.file.path)

    bgrHist = clrHistogram(img, None, [8, 8, 8])

    # extract features & save to db
    pic.bgrHist = bgrHist.tostring()
    pic.hsvHist = colour_extractor(img).tostring()

    pic.texture = texture_extractor(img).tostring()

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = LocalBinaryPatterns(20, 1)
    pic.lbpHist = lbp.describe(gimg).tostring()

    pic.save()

    return get_results(Image.objects.get(file=pic.file))


def colour_extractor(image):
    features = []
    weights = [0.4, 0.15, 0.15, 0.15, 0.15]
    hsvimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for mask in splitImage(hsvimg):
        features.append(clrHistogram(hsvimg, mask, [8, 12, 3]))

    return np.dot(np.array(weights, dtype=np.float32), np.array(features, dtype=np.float32))


def texture_extractor(image):
    props = ['ASM', 'contrast', 'correlation']
    features = []
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g = greycomatrix(grey, [2], [0, 90, 45, 135], 256, symmetric=True, normed=True)
    for p in props:
        v = greycoprops(g, p)
        var = np.var(v)
        u = np.mean(v)
        r = np.ptp(v)
        sd = np.std(v)
        features.append([u, r, sd, var])

    features = np.array(features)

    return features.flatten()


# get results sorted by similarity
def get_results(query):
    # construct dictionary of image path : distance
    w = [.4,.5,.1]
    s = Searcher(query)
    c = s.colour()
    t = s.lbpatterns()
    h = s.texture()
    matches = combine(c, t, h, w)

    dict_sorted = sorted([(v, k) for (k, v) in matches.items()])

    return dict_sorted[:8]


def combine(a, b, c, w):
    matches = {}

    # split dictionaries into keys and values
    al = [x for x in a.items()]
    ak, av = zip(*al)
    bl = [x for x in b.items()]
    bk, bv = zip(*bl)
    cl = [x for x in c.items()]
    ck, cv = zip(*cl)

    # scale the values in the range 0-1
    a_scaled = preprocessing.minmax_scale(av, feature_range=(0, 1))
    b_scaled = preprocessing.minmax_scale(bv, feature_range=(0, 1))
    c_scaled = preprocessing.minmax_scale(cv, feature_range=(0, 1))

    # build numpy structured arrays combining scaled values and original keys
    names = ['keys', 'values']
    formats = ['S225', 'f8']
    dtype = dict(names=names, formats=formats)
    anp = np.array(list(zip(ak,a_scaled)), dtype=dtype)
    bnp = np.array(list(zip(bk,b_scaled)), dtype=dtype)
    cnp = np.array(list(zip(ck,c_scaled)), dtype=dtype)

    # iterate over numpy structures creating a weighted average between values with the same key
    for i, t1 in np.ndenumerate(anp):
        for j, t2 in np.ndenumerate(bnp):
            if anp['keys'][i] == bnp['keys'][j]:
                for k, t3 in np.ndenumerate(cnp):
                    if anp['keys'][i] == cnp['keys'][k]:
                        stack = np.vstack((anp['values'][i], bnp['values'][j]))
                        stack = np.vstack((stack, cnp['values'][k]))
                        matches[anp['keys'][i].decode("utf-8")] = np.average(stack, axis=0, weights=w)[0]
                        break
                break

    return matches


# create normalize and flatten a color histogram
def clrHistogram(image, mask, bins):
    hist = cv2.calcHist([image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()


# split the given image into 5 sections top-left, top-right, center, bottom-left, bottom-right & return the masks
def splitImage(image):
    masks = []

    # grab the dimensions and compute the center of the image
    (h, w) = image.shape[:2]
    (cX, cY) = (int(w * 0.5), int(h * 0.5))

    # divide the image into four rectangles/segments (top-left,
    # top-right, bottom-right, bottom-left)
    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

    # construct an elliptical mask representing the center of the image
    (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
    ellipMask = np.zeros(image.shape[:2], dtype="uint8")
    # ellipse(image, centre, axes, rotation angle, start angle, end angle, colour, thickness)
    cv2.ellipse(ellipMask, (int(cX), int(cY)), (int(axesX), int(axesY)), 0, 0, 360, 255, -1)
    masks.append(ellipMask)

    # loop over the segments
    for (startX, endX, startY, endY) in segments:
        # construct a mask for each corner of the image, subtracting
        # the elliptical center from it
        cornerMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        masks.append(cv2.subtract(cornerMask, ellipMask))

    return masks


def reload(request):
    arr = []

    for instance in Image.objects.all():
        img = cv2.imread(instance.file.path)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = LocalBinaryPatterns(20, 1)
        instance.texture = lbp.describe(grey).tostring()
        # instance.hsvHist = colour_extractor(img).tostring()
        instance.save()
        arr.append(str(instance.file) + ' :: reprocessed<br>')

    return HttpResponse(arr)


# debug variable
def pdb(element):
    import pdb; pdb.set_trace()
    return element
