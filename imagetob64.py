"""
Licensed under Public Domain Mark 1.0. 
See http://creativecommons.org/publicdomain/mark/1.0/
Author: Justin Bruce Van Horne <justinvh@gmail.com>
"""


"""
Python-Markdown base-64 encoder
"""

from sys import version
import re
import os
import string
import base64
import markdown

# Defines our basic inline image
IMG_EXPR = "<img class='%s' alt='%s' id='%s'" + \
        " src='data:image/png;base64,%s'>"

# Base CSS template
IMG_CSS = \
        "<style scoped>img-inline { vertical-align: middle; }</style>\n"

regexp = re.compile("^!\[(.*)\]\((.*)\)$")

class ImagePreprocessor(markdown.preprocessors.Preprocessor):

    def run(self, lines):
        """Parses the actual page"""
        # Re-creates the entire page so we can parse in a multine env.
        page = "\n".join(lines)
        images = regexp.findall(page)

        # No sense in doing the extra work
        if not len(images):
            return page.split("\n")

        for i in images:
            png = open(i[1], "rb")
            data = png.read()
            data = base64.b64encode(data)
            png.close()
            page = reg.sub(IMG_EXPR %
                    (data), page, 1)
            pass # do stuff with page

        # Make sure to resplit the lines
        return page.split("\n")

class ImagePostprocessor(markdown.postprocessors.Postprocessor):
    """This post processor extension just allows us to further
    refine, if necessary, the document after it has been parsed."""
    def run(self, text):
        # Inline a style for default behavior
        text = IMG_CSS + text
        return text

class MarkdownImage(markdown.Extension):
    """Wrapper for Preprocessor"""
    def extendMarkdown(self, md, md_globals):
        # Our base LaTeX extension
        md.preprocessors.add('image-to-b64',
                Imagereprocessor(self), ">html_block")

def makeExtension(*args, **kwargs):
    """Wrapper for a MarkDown extension"""
    return MarkdownImage(*args, **kwargs)
