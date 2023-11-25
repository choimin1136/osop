import selectivesearch
import os

def ss(img):
    # Selective Search를 통해 region proposal 수행
    _, regions = selectivesearch.selective_search(img, scale=100, min_size=2000)
    rects = [cand['rect'] for cand in regions]
    return rects