__version__ = "0.1"
__all__ = ["SimpleHTTPRequestHandler"]
 
import os
import posixpath
import http.server
import urllib.request, urllib.parse, urllib.error
import cgi
import shutil
import mimetypes
import re
import math
from io import BytesIO
import dlib
from skimage.feature import hog
from skimage import data, exposure
from matplotlib import pyplot
from requests_toolbelt.multipart import decoder
import time


class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
 
    face_descriptors = {}

    step_times = []

    labeled_chips = {}

    hog_face_detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    trained_detector = None

    def do_GET(self):
        f = self.send_head()
        if f:
            self.copyfile(f, self.wfile)
            f.close()
 
    def do_HEAD(self):
        f = self.send_head()
        if f:
            f.close()
 
    def do_POST(self):
        r, info = self.deal_post_data()
        if r:
            return
        print((r, info, "by: ", self.client_address))
        f = BytesIO()
        f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write(b"<html>\n<title>Upload Result Page</title>\n")
        f.write(b"<body>\n<h2>Upload Result Page</h2>\n")
        f.write(b"<hr>\n")
        f.write(b"<strong>Failed:</strong>")
        f.write(info.encode())
        f.write(("<br><a href=\"%s\">back</a>" % self.headers['referer']).encode())
        f.write(b"</body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if f:
            self.copyfile(f, self.wfile)
            f.close()
        
    def deal_post_data(self):
        content_type = self.headers['content-type']
        data_length = int(self.headers['content-length'])
        if not content_type:
            return (False, "No Content-Type header")
        multipart_data = decoder.MultipartDecoder(self.rfile.read(data_length), content_type)
        filename = None
        label = None
        mark = {}
        for part in multipart_data.parts:
            content_disposition = part.headers[b"Content-Disposition"]
            if (b"name=\"min" in content_disposition or b"name=\"max" in content_disposition):
                mark[re.findall("name=\"(\w*?)\"", content_disposition.decode())[0]] = part.text
            if (b"name=\"filename\"" in content_disposition):
                filename = part.text
            if b"name=\"file\"" in content_disposition:
                filename = re.findall(r'.*filename="(.*)"', content_disposition.decode())[0]
                print("filename is " + filename)
                file_content = part.content
            if b"name=\"action\"" in content_disposition:
                action = part.text
                print("action is " + action)
            if b"name=\"label\"" in content_disposition:
                label = part.text
                print("label is " + label)

        if len(mark) > 0 and "minX" in mark and mark["minX"]:
            trainpath = os.path.join(self.translate_path(self.path), "train.txt")
            try:
                out = open(trainpath, 'a')
                out.write(str(mark["minX"]) + ";" +str(mark["minY"]) + ";" + str(mark["maxX"]) + ";" + str(mark["maxY"]) + ";" + filename + "\n")
                out.close()
            except IOError:
                return (False, "Can't create file %s to write" % trainpath)
            self.display_success(None, "appended Marker for %s to training data" % filename)
            return (True, "success")
        if not action:
            return (False, "Can't find out action")
        if (action == "resetTraining"):
            trainpath = os.path.join(self.translate_path(self.path), "train.txt")
            os.remove(trainpath)
            self.display_success(None, "resetted training data")
            return (True, "success")
        if (action == "doTraining"):

            self.display_success(None, "Training successful")
            return (True, "success")
        if not filename:
            return (False, "Can't find out file name")
        if (filename.endswith(".py") or filename.endswith(".dat")):
             directory = self.translate_path(self.path)
        else:
            if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
                return (False, "Unknown File type (not png, jpeg or jpg)")
            filename = filename.lower()
            path = self.translate_path(self.path)
            directory = os.path.join(path, os.path.splitext(filename)[0])
            if not os.path.exists(directory):
                os.makedirs(directory)
        absolutefile = os.path.join(directory, filename)
        try:
            out = open(absolutefile, 'wb')
        except IOError:
            return (False, "Can't create file %s to write" % absolutefile)
                
        out.write(file_content)
        out.close()
        self.process_uploaded_file(absolutefile, filename, action, label)
        return (True, "File '%s' upload success!" % filename)

    def process_uploaded_file(self, absoluteFile: str, filename:str, action: str, label: str):
        self.step_times.clear()
        self.labeled_chips.clear()
        if (action == "upload"):
            self.display_success(filename, "Successfully uploaded file ")
        if (action == "hog"):
            self.create_hog_image(absoluteFile, filename)
            self.display_files(filename, "hog")
        if (action == "facedetection"):
            self.detect_faces(absoluteFile, filename)
            self.display_files(filename, "facedetection")
        if (action == "facelandmark"):
            self.landmark_faces(absoluteFile, filename)
            self.display_files(filename, "facelandmark")
        if (action == "facelabel"):
            self.label_face(absoluteFile, filename, label)
            self.display_files(filename, label)
        if (action == "facerecognition"):
            self.recognize_face(absoluteFile, filename)
            self.display_files(filename, "facerecognition")
        if (action == "markregion"):
            self.display_files(filename, None)
        if (action == "useTrainedDetector"):
            self.use_trained_detector(absoluteFile, filename)
            self.display_files(filename, "trainedhog")

    def create_hog_image(self, absoluteFile: str, filename: str):
        image = dlib.load_grayscale_image(absoluteFile)
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
        start_time = time.clock()
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        self.step_times.append("HOG Creation: %.3f s" % (time.clock() - start_time))
        processed_file = self.path_to_processed_file(filename, "hog")
        pyplot.imsave(processed_file, hog_image_rescaled)

    def detect_faces(self, absoluteFile: str, filename: str):
        image = dlib.load_rgb_image(absoluteFile)
        start_time = time.clock()
        face_boundaries = self.hog_face_detector(image, 1)
        self.step_times.append("HOG + SVM Detection: %.3f s" % (time.clock() - start_time))
        for i, d in enumerate(face_boundaries):
            draw_rectangle(image, d)
        processed_file = self.path_to_processed_file(filename, "facedetection")
        dlib.save_image(image, processed_file)

    def landmark_faces(self, absoluteFile: str, filename: str):
        image = dlib.load_rgb_image(absoluteFile)
        start_time = time.clock()
        dets_hog = self.hog_face_detector(image, 1)
        self.step_times.append("HOG + SVM Detection: %.3f s" % (time.clock() - start_time))
        for i, d in enumerate(dets_hog):
            start_time = time.clock()
            shape = self.predictor(image, d)
            self.step_times.append("Landmark Detection: %.3f s" % (time.clock() - start_time))
            draw_marker(image, shape.parts())
        processed_file = self.path_to_processed_file(filename, "facelandmark")
        dlib.save_image(image, processed_file)

    def recognize_face(self, absoluteFile: str, filename: str):
        if len(self.face_descriptors) == 0:
            return False
        image = dlib.load_rgb_image(absoluteFile)
        start_time = time.clock()
        dets_hog = self.hog_face_detector(image, 1)
        self.step_times.append("HOG + SVM Detection: %.3f s" % (time.clock() - start_time))
        for i, d in enumerate(dets_hog):
            start_time = time.clock()
            shape = self.predictor(image, d)
            self.step_times.append("Landmark Detection: %.3f s" % (time.clock() - start_time))

            face_chip = dlib.get_face_chip(image, shape)
            chip_file = self.path_to_processed_file(filename, str(i))
            dlib.save_image(face_chip, chip_file)

            start_time = time.clock()
            face_descriptor = self.facerec.compute_face_descriptor(image, shape)
            self.step_times.append("Face Descriptor Computation:  %.3f s" % (time.clock() - start_time))

            draw_rectangle(image, d)
            face_label = "???[" + str(i) + "]"
            for label, known_descriptor in self.face_descriptors.items():
                distance = 0.0
                for i in range(len(face_descriptor)):
                    distance = distance + (face_descriptor[i] - known_descriptor[i]) * (face_descriptor[i] - known_descriptor[i])
                distance = math.sqrt(distance)
                if (distance < 0.6):
                    face_label = label + (" (distance=%.3f)" % distance)
                print("Distance to {} is {}".format(label, distance))
            self.labeled_chips[face_label] = chip_file
        processed_file = self.path_to_processed_file(filename, "facerecognition")
        dlib.save_image(image, processed_file)

    def label_face(self, absoluteFile: str, filename: str, label: str):
        if not label:
            return False
        image = dlib.load_rgb_image(absoluteFile)
        start_time = time.clock()
        dets_hog = self.hog_face_detector(image, 1)
        self.step_times.append("HOG + SVM Detection: %.3f s" % (time.clock() - start_time))
        if len(dets_hog) != 1:
            return False
        for i, d in enumerate(dets_hog):
            start_time = time.clock()
            shape = self.predictor(image, d)
            self.step_times.append("Landmark Detection: %.3f s" % (time.clock() - start_time))

            start_time = time.clock()
            face_descriptor = self.facerec.compute_face_descriptor(image, shape)
            self.step_times.append("Face Descriptor Computation: %.3f s" % (time.clock() - start_time))

            self.face_descriptors[label] = face_descriptor;
            draw_rectangle(image, d)
        print("labeled face as " + label)
        processed_file = self.path_to_processed_file(filename, label)
        dlib.save_image(image, processed_file)

    def do_training(self):
        trainpath = os.path.join(self.translate_path(self.path), "train.txt")
        trainfile = open(trainpath, 'r')
        testdatadef = trainfile.readlines()
        trainfile.close()

        images = []
        boxes = []
        for testdata in testdatadef:
            testvalues = testdata.split(";")
            boxes.append([dlib.rectangle(int(testvalues[0]), int(testvalues[1]), int(testvalues[2]), int(testvalues[3]))])
            imagepath = os.path.join(self.translate_path(self.path), testvalues[4].strip("\n"))
            images.append(dlib.load_grayscale_image(imagepath))
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = True
        options.C = 5
#        options.be_verbose = False
        SimpleHTTPRequestHandler.trained_detector = dlib.train_simple_object_detector(images, boxes, options)

    def use_trained_detector(self, absoluteFile: str, filename: str):
        print(type(SimpleHTTPRequestHandler.trained_detector))
        image = dlib.load_rgb_image(absoluteFile)
        start_time = time.clock()
        dets_hog = SimpleHTTPRequestHandler.trained_detector(image, 1)
        self.step_times.append("Trained HOG + SVM Detection: %.3f s" % (time.clock() - start_time))
        for i, d in enumerate(dets_hog):
            print("detected object: " + str(d))
            draw_rectangle(image, d)
        processed_file = self.path_to_processed_file(filename, "trainedhog")
        dlib.save_image(image, processed_file)

    def display_files(self, filename, classifier):
        uploaded_file = self.path_to_uploaded_file(filename)
        if classifier != None:
            processed_file = self.path_to_processed_file(filename, classifier)
        f = BytesIO()
        f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write(b"<html>\n<title>Processed File result</title>\n")
        if classifier != None:
            f.write(("<img src=\"%s\"/>\n" % uploaded_file).encode())
            f.write(("<img src=\"%s\"/>\n" % processed_file).encode())
        else:
            f.write(b"<form ENCTYPE=\"multipart/form-data\" method=\"post\"\n>")
            f.write(b"<div id=\"selector\" hidden=\"hidden\" style=\"border: 3px dotted red; position: absolute;\" ")
            f.write(b"<div id=\"selector\" hidden=\"hidden\" style=\"border: 3px dotted red; position: absolute;\" ")
            f.write(b"onmousedown=\"startMarker(event)\" onmousemove=\"updateMarker(event)\" onmouseup=\"finalizeMarker(event)\"></div>")
            f.write(b"<input id=\"minX\" name=\"minX\" type=\"hidden\" /><input id=\"maxX\" name=\"maxX\" type=\"hidden\" />")
            f.write(b"<input id=\"minY\" name=\"minY\" type=\"hidden\" /><input id=\"maxY\" name=\"maxY\" type=\"hidden\" />")
            f.write(("<input name=\"filename\" type=\"hidden\" value=\"%s\"/>"% uploaded_file).encode())
            f.write(b"<div id=\"selector\" hidden=\"hidden\" style=\"border: 1px dotted #000; position: absolute;\"></div>")
            f.write(b"<script>var x1, x2, y1, y2; var doSelect=false;\n")
            f.write(b"function updateSelector() { minX.value = Math.min(x1,x2); maxX.value = Math.max(x1,x2); minY.value = Math.min(y1,y2); maxY.value = Math.max(y1,y2); ")
            f.write(b"selector.style.left = minX.value + 'px'; selector.style.top = minY.value + 'px';")
            f.write(b"selector.style.width = (maxX.value - minX.value) + 'px'; selector.style.height = (maxY.value - minY.value) + 'px'; };\n")
            f.write(b"function startMarker(e) { doSelect=true; selector.hidden = 0; x1 = e.clientX; y1 = e.clientY; updateSelector();debugEl.textContent=\'started\';};\n")
            f.write(b"function updateMarker(e) {if (doSelect) {x2 = e.clientX; y2 = e.clientY; updateSelector(); debugEl.textContent= '(' + x1 +\',\' + y1 + '):(' + x2 +\',\' + y2 + \')\'}};\n")
            f.write(b"function finalizeMarker(e) {updateMarker(e); doSelect=false; debugEl.textContent=\'stopped\';};</script>\n")
            f.write(("<img src=\"%s\" draggable=\"false\" " % uploaded_file).encode())
            f.write(b"onmousedown=\"startMarker(event)\" onmousemove=\"updateMarker(event)\" onmouseup=\"finalizeMarker(event)\"/><span id=\"debugEl\" ></span><br/>")
            f.write(b"<input type=\"submit\" name=\"mark\" value=\"Mark region\"/></form>\n")
            f.write(b"</form>\n")
        if (len(self.labeled_chips) > 0):
            f.write(b"<h2>recognition result</h2>\n<table style=\"width:100%\"><tr>\n")
            for label, chip_file in self.labeled_chips.items():
                f.write(("<td><img src=\"%s\"/></td>\n" % chip_file).encode())
            f.write(b"</tr><tr>")
            for label, chip_file in self.labeled_chips.items():
                f.write(("<td>%s</td>\n" % label).encode())
            f.write(b"</tr></table><br><br>\n")
        f.write(b"<h2>process times</h2>")
        for step_time in self.step_times:
            f.write((step_time + "<br>").encode())
        f.write(("<br><a href=\"%s\">back</a>" % self.headers['referer']).encode())
        f.write(b"</body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self.copyfile(f, self.wfile)
        f.close()

    def display_success(self, filename: str, message: str):
        f = BytesIO()
        f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write(b"<html>\n<title>Upload Result Page</title>\n")
        f.write(("<strong>%s</strong>" % message).encode())
        if filename:
            f.write(filename.encode())
        f.write(("<br><a href=\"%s\">back</a>" % self.headers['referer']).encode())
        f.write(b"</body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if f:
            self.copyfile(f, self.wfile)
            f.close()


    def path_to_processed_file(self, filename, classifier):
        return os.path.splitext(filename)[0] + '/' + os.path.splitext(filename)[0] + '_' + classifier + os.path.splitext(filename)[1]

    def path_to_uploaded_file(self, filename):
        return os.path.splitext(filename)[0] + '/' + filename

    def send_head(self):
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            if not self.path.endswith('/'):
                # redirect browser - doing basically what apache does
                self.send_response(301)
                self.send_header("Location", self.path + "/")
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    break
            else:
                return self.list_directory(path)
        ctype = self.guess_type(path)
        try:
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(path, 'rb')
        except IOError:
            self.send_error(404, "File not found")
            return None
        self.send_response(200)
        self.send_header("Content-type", ctype)
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f
 
    def list_directory(self, path):
        try:
            list = os.listdir(path)
        except os.error:
            self.send_error(404, "No permission to list directory")
            return None
        list.sort(key=lambda a: a.lower())
        f = BytesIO()
        displaypath = cgi.escape(urllib.parse.unquote(self.path))
        f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write(("<html>\n<title>Directory listing for %s</title>\n" % displaypath).encode())
        f.write(("<body>\n<h2>Directory listing for %s</h2>\n" % displaypath).encode())
        f.write(b"<hr>\n")
        f.write(b"<form ENCTYPE=\"multipart/form-data\" method=\"post\">")
        f.write(b"<input name=\"file\" accept=\"image/*;capture=camera\" type=\"file\"/>")
        f.write(b"<select name=\"action\" />")
        f.write(b"<option value=\"upload\">Upload</option>")
        f.write(b"<option value=\"hog\">Generate HOG</option>")
        f.write(b"<option value=\"facedetection\">Detect faces</option>")
        f.write(b"<option value=\"facelandmark\">Landmark faces</option>")
        f.write(b"<option value=\"facelabel\">Label a face</option>")
        f.write(b"<option value=\"facerecognition\">Recognize faces</option>")
        f.write(b"<option value=\"markregion\">Mark region</option>")
        f.write(b"<option value=\"resetTraining\">Reset training data</option>")
        f.write(b"<option value=\"doTraining\">Train HOG detector with training data</option>")
        f.write(b"<option value=\"useTrainedDetector\">Use trained HOG Detector</option>")
        f.write(b"</select/>")
        f.write(b" Label: <input name=\"label\"/>")
        f.write(b"<input type=\"submit\" value=\"process\"/></form>\n")
        f.write(b"<hr>\n<ul>\n")
        for name in list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
            if os.path.islink(fullname):
                displayname = name + "@"
                # Note: a link to a directory displays with @ and links with /
            f.write(('<li><a href="%s">%s</a>\n'
                    % (urllib.parse.quote(linkname), cgi.escape(displayname))).encode())
        f.write(b"</ul>\n<hr>\n</body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f
 
    def translate_path(self, path):
        # abandon query parameters
        path = path.split('?',1)[0]
        path = path.split('#',1)[0]
        path = posixpath.normpath(urllib.parse.unquote(path))
        words = path.split('/')
        words = [_f for _f in words if _f]
        path = os.getcwd()
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir): continue
            path = os.path.join(path, word)
        return path

    def copyfile(self, source, outputfile):
        shutil.copyfileobj(source, outputfile)
 
    def guess_type(self, path):
        base, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']

    if not mimetypes.inited:
        mimetypes.init() # try to read system mime.types
    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream', # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
        })

def test(HandlerClass = SimpleHTTPRequestHandler,
         ServerClass = http.server.HTTPServer):
    http.server.test(HandlerClass, ServerClass)

def draw_rectangle(img, rect):
    markerpixel = [255,0,0]
    for x in range(rect.left(), rect.right()):
        img[rect.top()][x] = markerpixel
        img[rect.top() + 1][x] = markerpixel
        img[rect.top() - 1][x] = markerpixel
        img[rect.bottom()][x] = markerpixel
        img[rect.bottom() - 1][x] = markerpixel
        img[rect.bottom() + 1][x] = markerpixel
    for y in range(rect.top(), rect.bottom()):
        img[y][rect.left()] = markerpixel
        img[y][rect.left() - 1] = markerpixel
        img[y][rect.left() + 1] = markerpixel
        img[y][rect.right()] = markerpixel
        img[y][rect.right() - 1] = markerpixel
        img[y][rect.right() + 1] = markerpixel

def draw_marker(img, markers):
    markerpixel = [0,255,0]
    for point in enumerate(markers):
        for i in range(-6, 7): # line lenght
            for j in range(-1, 2): # line width
                img[point[1].y + j][point[1].x + i] = markerpixel
                img[point[1].y + i][point[1].x + j] = markerpixel

if __name__ == '__main__':
    test()