# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:59:37 2017

@author: Dani
"""

from __future__ import division

import pdfminer
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import LAParams, LTChar
from pdfminer.converter import PDFPageAggregator

from scipy.signal import medfilt

from glob import glob
from os import path, makedirs, remove
from sys import argv
import re

import types
from copy import deepcopy

from collections import deque
from random import sample

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages

from pyprind import ProgBar as pb

from PyPDF2 import PdfFileReader, PdfFileMerger

import pandas as pd

np = pd.np

TEXT_ELEMENTS = [
    pdfminer.layout.LTTextBox,
    pdfminer.layout.LTTextBoxHorizontal,
    pdfminer.layout.LTTextLine,
    pdfminer.layout.LTTextLineHorizontal
]

CONTAINERS = [
    pdfminer.layout.LTPage,
    pdfminer.layout.LTFigure
]


def try_join_adjacent_charboxes(pretenders, mode, pbar=None):
    
    which = []
    exclude = []
    any_ = False
    charboxes = list(pretenders)
    
    for i in range(len(pretenders)):

        if i in exclude:
            continue

        for j in [j for j in range(len(charboxes)) if j != i and j not in which + exclude]:
            if charboxes[j].maybe_join(pretenders[i], mode=mode):

                which.append(i)
                exclude.append(j)
                any_ = True

                if pbar is not None:
                    pbar.update()

                break

        if pbar is not None:
            pbar.update()
    
    [charboxes.pop(i) for i in reversed(which)]
    
    return charboxes, any_


def try_join_all_adjacent_charboxes(pretenders, mode='H'):

    any_ = True
    while any_:

        qty = len(pretenders)
        pbar = None
        if qty > 1000:
            print('')
            pbar = pb(len(pretenders))

        pretenders, any_ = try_join_adjacent_charboxes(pretenders, mode, pbar=pbar)

    return pretenders


def filter_non_content_compliant(pretenders, regexp):

    which = []

    for i in range(len(pretenders)):
        if re.match(regexp, pretenders[i]._text) is None:
            which.append(i)

    [pretenders.pop(i) for i in reversed(which)]

    return pretenders


def filter_non_table_eligible(pretenders):

    which = []
    exclude = []
    groupings = []
    qty = len(pretenders)

    med_height = np.median(retrieve_heights(pretenders))

    for i in range(qty):

        if i in exclude:
            continue

        any_ = False
        pret1 = pretenders[i]

        for j in range(qty):

            if j == i:
                continue

            pret2 = pretenders[j]
            if abs((pret1.y0 + pret1.y1)/2 - (pret2.y0 + pret2.y1)/2) < med_height/3:

                any_ = True
                exclude.append(j)

                was_mapped_ = False
                for g in groupings:
                    if j in g['elements']:
                        g['elements'].append(i)
                        g['coordinates'].append((pret1.x0, pret1.y0, pret1.x1, pret1.y1))
                        was_mapped_ = True

                if not was_mapped_:
                    groupings.append(
                        dict(elements=[i, j],
                             coordinates=[(pret1.x0, pret1.y0, pret1.x1, pret1.y1),
                                          (pret2.x0, pret2.y0, pret2.x1, pret2.y1)])
                    )

                break

        if not any_:
            which.append(i)

    [pretenders.pop(i) for i in reversed(which)]

    return pretenders, groupings


def retrieve_heights(pretenders):
    return [pret.y1 - pret.y0 for pret in pretenders]


def retrieve_widths(pretenders):
    return [pret.x1 - pret.x0 for pret in pretenders]


def retrieve_vertical_coordinates(pretenders):
    return [(pret.y0, pret.y1) for pret in pretenders]


def retrieve_horizontal_coordinates(pretenders):
    return [(pret.x0, pret.x1) for pret in pretenders]


def calculate_vertical_projection_histogram(pretenders, max_y):

    vcs = retrieve_vertical_coordinates(pretenders)

    hist = np.zeros((round(max_y),))

    for p in pretenders:
        for v in range(int(p.y0), int(p.y1)+1):
            hist[v] += 1

    return hist


def calculate_horizontal_projection_histogram(pretenders, max_x):

    vcs = retrieve_horizontal_coordinates(pretenders)

    hist = np.zeros((round(max_x),))

    for p in pretenders:
        for v in range(int(p.x0), int(p.x1)+1):
            hist[v] += 1

    return hist


def flatten(lst):
    """Flattens a list of lists"""
    return [subelem for elem in lst for subelem in elem]


def uncontainerize(lst):
    """Iteratively flattens a list"""
    result = []

    if any([isinstance(lst, t) for t in [list] + CONTAINERS]):
        for el in lst:
            result.extend(uncontainerize(el))
    else:
        result.append(lst)

    return result


def draw_rect_bbox(coords, ax_, color):
    """
    Draws an unfilled rectable onto ax.
    """
    x0, y0, x1, y1 = coords
    ax_.add_patch(
        patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            color=color
        )    
    )

    
def draw_rect(rect_, ax_, color="black"):
    draw_rect_bbox((rect_.x0, rect_.y0, rect_.x1, rect_.y1), ax_, color)


def draw_text(cb, ax_, page_size):
    fontsize = int(.9 * cb.size * float(page_size)/15)
    ax_.text(cb.x0 + int(fontsize/4), cb.y0 + int(fontsize/4), cb._text, fontsize=fontsize)


def extract_layout_by_page(pdf_path):
    """
    Extracts LTPage objects from a pdf file.
    
    slightly modified from
    https://euske.github.io/pdfminer/programming.html
    """
    laparams = LAParams()
    laparams.detect_vertical = True

    fp = open(pdf_path, 'rb')
    parser = PDFParser(fp)
    document = PDFDocument(parser)

    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed

    rsrcmgr = PDFResourceManager()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    layouts = []
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
        layouts.append(device.get_result())

    return layouts

    
def maybe_join(self, obj, mode='H'):

    self_x_cent = (self.x0 + self.x1) / 2
    obj_x_cent = (obj.x0 + obj.x1) / 2

    self_y_cent = (self.y0 + self.y1) / 2
    obj_y_cent = (obj.y0 + obj.y1) / 2

    if (((mode == 'H') and (abs(self_y_cent - obj_y_cent) < .1) and (self.hdistance(obj) < 1.)) or
            ((mode == 'V') and (self.vdistance(obj) < .1) and self.is_hoverlap(obj))):

        self.x0 = min(self.x0, obj.x0)
        self.x1 = max(self.x1, obj.x1)
        self.y0 = min(self.y0, obj.y0)
        self.y1 = max(self.y1, obj.y1)

        if ((mode == 'H' and obj_x_cent > self_x_cent) or
                (mode == 'V' and obj_y_cent > self_y_cent)):
            if mode == 'V':
                self._text = self._text.strip() + ' ' + obj._text.strip()
            else:
                self._text += obj._text
        else:
            if mode == 'V':
                self._text = obj._text.strip() + ' ' + self._text.strip()
            else:
                self._text = obj._text + self._text

        return True

    return False


def extract_characters(element):
    """
    Recursively extracts individual characters from 
    text elements. 
    """
    if isinstance(element, LTChar):
        element.maybe_join = types.MethodType(maybe_join, element)
        return [element]

    if any(isinstance(element, t) for t in (TEXT_ELEMENTS + CONTAINERS + [list])):
        return flatten([extract_characters(l) for l in element])

    return []

dirname = path.dirname(path.realpath(__file__))
data_dir = path.join(dirname, 'data')
result_dir = path.join(dirname, 'results')
numpages = -1

if(len(argv)>1):
    data_dir = path.join(data_dir, argv[1])
	
if(len(argv)>2):
    numpages = int(argv[2])

formats = {
    'PDF': '*.pdf',
    'DOC': '*.doc',
    'DOCX': '*.docx',
    'XLS': '*.xls',
    'XLSX': '*.xlsx',
    'CSV': '*.csv',
    'TSV': '*.tsv',
    'JSON': '*.json',
    'TXT': '*.txt',
}

results = {name: dict(format=f, paths=[p for p in glob(path.join(data_dir, f))]) for name, f in formats.items()}

if not path.exists(result_dir):
    makedirs(result_dir)

for p in results['PDF']['paths']:

    p_result = path.join(result_dir, path.split(p)[-1])
    pp = PdfPages(p_result)

    p_layouts = extract_layout_by_page(p)
    if(numpages >= 0):
        numpages = min(numpages, len(p_layouts))
        p_layouts = sample(p_layouts, numpages)

    for p_layout in p_layouts:

        texts = []
        rects = []

        # separate text and rectangle elements
        for e in uncontainerize(p_layout):
            if any([isinstance(e, t) for t in [pdfminer.layout.LTTextBoxHorizontal, pdfminer.layout.LTChar]]):
                texts.append(e)
            elif isinstance(e, pdfminer.layout.LTRect):
                rects.append(e)

        # sort them into
        characters = extract_characters(texts)

        xmin, ymin, xmax, ymax = p_layout.bbox
        size = 9

        h_adj_chrbxs = try_join_all_adjacent_charboxes(characters, 'H')
        h_adj_chrbxs = filter_non_content_compliant(h_adj_chrbxs, r"\s*\S+")

        h_hist = calculate_horizontal_projection_histogram(h_adj_chrbxs, xmax)
        h_hist = medfilt(h_hist, [5])

        fig, ax = plt.subplots(figsize=(size, size * (h_hist.max() / xmax)))
        ax.bar(np.arange(0, h_hist.size, 1.), h_hist*size, .9)
        pp.savefig()
        plt.close()

        v_hist = calculate_vertical_projection_histogram(h_adj_chrbxs, ymax)

        p_layout

        t_eli_chrbxs, groupings = filter_non_table_eligible(deepcopy(h_adj_chrbxs))
        #adj_chrbxs = try_join_all_adjacent_charboxes(h_adj_chrbxs, 'V')

        g_boxes = []
        for g in groupings:
            x0, y0, x1, y1 = g['coordinates'][0]
            for c in g['coordinates'][1:]:

                x0 = min(c[0], x0)
                y0 = min(c[1], y0)
                x1 = max(c[2], x1)
                y1 = max(c[3], y1)

            g_boxes.append((x0, y0, x1, y1))

        fig, ax = plt.subplots(figsize=(size, size * (ymax / xmax)))

        #for rect in rects:
        #    draw_rect(rect, ax)

        for g in g_boxes:
            draw_rect_bbox(g, ax, "green")

        for c in h_adj_chrbxs:
            draw_rect(c, ax, "red")

        for c in t_eli_chrbxs:
            draw_text(c, ax, size)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        pp.savefig()

        plt.close()

    pp.close()

    #with open(p, 'rb') as fp1:
    #
    #    with open(p_result, 'rb') as fp2:
    #
    #        output = PdfFileWriter()
    #        pdfOne = PdfFileReader(fp1)
    #        pdfTwo = PdfFileReader(fp2)
    #
    #        for i in range(pdfOne.getNumPages()):
    #            output.addPage(pdfOne.getPage(i))
    #
    #        for i in range(pdfTwo.getNumPages()):
    #            output.addPage(pdfTwo.getPage(i))
    #
    #        with open(path.splitext(p_result)[0] + '_final.pdf', "wb") as outputStream:
    #            output.write(outputStream)

    merger = PdfFileMerger(strict=False)
    merger.append(PdfFileReader(p, 'rb'))
    merger.append(PdfFileReader(p_result, 'rb'))
    merger.write(path.splitext(p_result)[0] + '_final.pdf')

    remove(p_result)
