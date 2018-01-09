from os import remove, path, getcwd
from pyPdf import PdfFileReader, PdfFileWriter
from tempfile import NamedTemporaryFile
from PythonMagick import Image
import argparse


def convert(args):

    dirname = getcwd()
    ifilename = path.join(dirname, args.ifile) if not path.isabs(args.ifile) else args.ifile
    ofilename = path.join(dirname, args.ofile) if not path.isabs(args.ofile) else args.ofile
    ofilename_n_ext = path.splitext(ofilename)[0]
    
    reader = PdfFileReader(open(ifilename, "rb"))
    for page_num in xrange(reader.getNumPages()):
        writer = PdfFileWriter()
        writer.addPage(reader.getPage(page_num))
        with open(path.join(dirname, 'temp.pdf'), 'wb') as temp:
            writer.write(temp)

        im = Image()
        im.density("300")# DPI, for better quality
        im.backgroundColor('white')
        im.fillColor('white')
        im.read(path.join(dirname, 'temp.pdf'))
        im.write("%s_%d.jpg" % (ofilename_n_ext, page_num))
        
        remove(path.join(dirname, 'temp.pdf'))
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('ifile', type=str, help="input_file")
    parser.add_argument('ofile', type=str, help="output_file")
    
    convert(parser.parse_args())
