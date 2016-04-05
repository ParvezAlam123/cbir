import os
import cv2
import json
import pickle
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.conf import settings
from sklearn.svm import LinearSVC
from skimage.feature import greycomatrix, greycoprops
from cbir.cvclasses.localbinarypatterns import LocalBinaryPatterns

from .forms import ImageForm
from .models import Image
from .models import Example
from .models import Classifier


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

        # train()

        pic.save()              # save image first so we can retrieve the file path

        data = process_image(pic)      # process the image; generate histograms etc

        return HttpResponse(json.dumps(data), content_type="application/json")

    return HttpResponseBadRequest('Image upload form not valid')


# extract features from query image
def process_image(pic):

    lbp = LocalBinaryPatterns(24, 8)
    clgfeatures = []

    img = cv2.imread(pic.file.path)
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extract a 3D RGB & HSV color histogram from the image using 8 bins per channel
    pic.bgrHist = clrHistogram(img, None, [8, 8, 8])

    for mask in splitImage(hsvimg):
        clgfeatures.extend(clrHistogram(hsvimg, mask, [18, 3, 3]))

    glcmfeatures = calcGLCM(grey)

    pic.lbpHist = lbp.describe(grey).tostring()
    pic.hsvHist = clgfeatures  # note that some data/precision is lost in the conversion float > string > float
    pic.dissimilarity = json.dumps(glcmfeatures['fdiss'])
    pic.correlation = json.dumps(glcmfeatures['fcorr'])
    pic.save()

    return searcher(pic)


# search bgrHist field and return 3 best matches
def searcher(pic):
    # construct dictionary of image path : distance
    matches = {}
    query = Image.objects.get(file=pic.file)
    w = [0.4, 0.6]

    for instance in Image.objects.all():
        if str(instance) != str(pic.file) and (instance.hsvHist and instance.dissimilarity and instance.correlation) is not None:
            texture1 = json.loads(query.dissimilarity) + json.loads(query.correlation)
            texture2 = json.loads(instance.dissimilarity) + json.loads(instance.correlation)
            x = [
                chi2_distance(toMatrix(query.hsvHist), toMatrix(instance.hsvHist)),
                chi2_distance(texture1, texture2)
            ]

            matches[str(instance)] = np.dot(w, x)

    dict_sorted = sorted([(v, k) for (k, v) in matches.items()])

    # svm = Classifier.objects.get(id=1).model
    # model = pickle.loads(svm)
    # pattern = np.fromstring(query.lbpHist)
    # prediction = model.predict(pattern)[0]
    # print(prediction)

    return dict_sorted[:8]


# create normalize and flatten a color histogram
def clrHistogram(image, mask, bins):
    hist = cv2.calcHist([image], [0, 1, 2], mask, bins, [0, 256, 0, 256, 0, 256])
    # TODO: does this need to be collapsed into one dimension? cv2.NormalizeHist(hist, 1)
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def chi2_distance(histA, histB, eps=1e-10):

    # temp3 = [item for item in histA if item not in histB]
    # d = {}
    # count = 1
    # for item in histA:
    #     if item not in histB:
    #         d[count] = type(item)
    #     count += 1
    #
    # pdb(d)

    # pdb(np.allclose(histA, histB))

    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

    return d


# converts string to numpy array (matrix). MUST CAST to float32 otherwise pythons float64 by default, this will not
# match with our query array even if they are the same image!!!
def toMatrix(text):
    text = text.replace('[', '').replace(']', '')
    floats = [np.float32(x) for x in text.split(',')]

    return floats


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
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        instance.texture = texture_extractor(gimg).tostring()
        instance.save()
        arr.append(str(instance.file) + ' :: reprocessed<br>')

    return HttpResponse(arr)


# trains a classifier and saves it in binary to the db
def train():
    lbp = LocalBinaryPatterns(24, 8)
    y = []  # list of responses
    x = []  # list of features

    training_dir = settings.MEDIA_ROOT + "\\images\\training"
    img_list = os.listdir(training_dir)

    for filename in img_list:
        # load image, convert to grey-scale and describe it using lbp
        img = cv2.imread(training_dir + "\\" + filename)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = lbp.describe(grey)

        # extract label from the image path
        y.append(filename.split("_")[-2])   # label
        x.append(hist)                      # vector
        Example(file=filename, lbpHist=hist.tostring(), label=filename.split("_")[-2]).save()

    model = LinearSVC(C=100.0, random_state=42)
    model.fit(x, y)     # train linear support vector machine
    Classifier(model=pickle.dumps(model)).save()

    # pdb(Classifier.objects.get(id=1))

    return


# debug variable
def pdb(element):
    import pdb; pdb.set_trace()
    return element
