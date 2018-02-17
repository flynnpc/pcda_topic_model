from matplotlib import pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import os

def mxWordCloud(cloudText):
    wrk_dir = os.path.dirname(__file__)
    imgMexico = os.path.join(wrk_dir, "mexico.png")
    mex_mask = np.array(Image.open(imgMexico))
    wordcloud = WordCloud(width = 640,
                        height = 480,
                        mask = mex_mask,
                        background_color="white").generate(cloudText)
    img = os.path.join(wrk_dir, "mxTopicCloud.jpg")
    wordcloud.to_file(img)
