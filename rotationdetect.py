import zipfile
import http.server
import socketserver
import os
import dlib
import glob
import shutil

class HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        
        if (self.path.startswith('/train')):
            self.unzip_post_body()
            self.train_object_detector()
            response ="training done"
            self.wfile.write(b"{\"training\" : \"done\"")
            shutil.rmtree('tmp')
            return
        if (self.path.startswith('/detect/zip')):
            self.unzip_post_body()
            self.use_object_detector()
            shutil.rmtree('tmp')
            return
        if (self.path.startswith('/detect/image')):
            self.write_image()
            self.use_object_detector()
            shutil.rmtree('tmp')
            return

    def do_GET(self):
        self.send_path_response()
    
    def send_path_response(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><head><title>Webservices</title></head>")
        self.wfile.write(b"<body><h1>Webservices</h1>")
        self.wfile.write(("<p>path: %s</p>" % self.path).encode())
        self.wfile.write(b"</body></html>")
        
    def unzip_post_body(self):
        content_type = self.headers['content-type']
        if not content_type:
            self.send_response (406, "No Content-Type header")
            return

        data_length_s = self.headers['content-length']
        if not data_length_s:
            self.send_response (406, "No Content-Length header")
            return
        data_length=int(data_length_s)
        
        post_body = self.rfile.read(data_length)
        
        tmp_zipfile_name = 'tmpfile.zip'
        tmp_zipfile = open(tmp_zipfile_name,'wb');
        tmp_zipfile.write(post_body)
        tmp_zipfile.close()
        
        with zipfile.ZipFile(tmp_zipfile_name, 'r') as zip_ref:
            zip_ref.extractall('tmp')
        os.remove(tmp_zipfile_name);
        return data_length
    
    def write_image(self):
        content_type = self.headers['content-type']
        if not content_type:
            self.send_response (406, "No Content-Type header")
            return

        data_length_s = self.headers['content-length']
        if not data_length_s:
            self.send_response (406, "No Content-Length header")
            return
        data_length=int(data_length_s)
        
        post_body = self.rfile.read(data_length)
        
        if (not os.path.isdir("tmp")):
            os.mkdir('tmp')
        tmp_image = open('tmp/image.png', 'wb');
        tmp_image.write(post_body)
        tmp_image.close()

        return data_length

    def train_object_detector(self):
        self.train_object_detector_for_path("landmarks.xml", "detector.svm")
        self.train_object_detector_for_path("landmarks_0.xml", "detector_0.svm")
        self.train_object_detector_for_path("landmarks_1.xml", "detector_1.svm")
        self.train_object_detector_for_path("landmarks_2.xml", "detector_2.svm")
        
    def train_object_detector_for_path(self, xmlPath, outputPath):
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = False
        options.C = 5
        options.num_threads = 4
        options.be_verbose = True
        
        training_xml_path = os.path.join('tmp', xmlPath)
        dlib.train_simple_object_detector(training_xml_path, outputPath, options)

    def use_object_detector(self):
        self.wfile.write(b"[")
        detector = dlib.simple_object_detector("detector.svm")
        firstFile = True
        for filename in glob.glob(os.path.join('tmp', "*.png")):
            print("Processing file: {}".format(filename))
            if (not firstFile):
                self.wfile.write(b",")
            self.wfile.write(("{\"image\":\"%s\",\"detections\":[" % filename).encode())

            image = dlib.load_rgb_image(filename)
            dets = detector(image)
            print("Number of objects detected: {}".format(len(dets)))
            firstDetection = True
            for i, d in enumerate(dets):
                print("detected object: " + str(d))
                if (not firstDetection):
                    self.wfile.write(b",")
                self.wfile.write(("{\"top\":%i,\"bottom\":%i,\"left\":%i,\"right\":%i}" % (d.top(), d.bottom(), d.left(), d.right())).encode())
                self.draw_rectangle(image, d)
                processed_file = self.path_to_processed_file(filename, "trainedhog")
                dlib.save_image(image, processed_file)
                firstDetection = False
            self.wfile.write(b"]}")
            firstFile = False
        self.wfile.write(b"]")

    def draw_rectangle(self, img, rect):
        markerpixel = [255,0,0]
        for x in range(rect.left(), rect.right()):
            self.point(x, rect.top(), img, markerpixel)
            self.point(x, rect.top() + 1, img, markerpixel)
            self.point(x, rect.top() - 1, img, markerpixel)
            self.point(x, rect.bottom(), img, markerpixel)
            self.point(x, rect.bottom() + 1, img, markerpixel)
            self.point(x, rect.bottom() - 1, img, markerpixel)
        for y in range(rect.top(), rect.bottom()):
            self.point(rect.left(), y, img, markerpixel)
            self.point(rect.left() - 1, y, img, markerpixel)
            self.point(rect.left() + 1, y, img, markerpixel)
            self.point(rect.right(), y, img, markerpixel)
            self.point(rect.right() - 1, y, img, markerpixel)
            self.point(rect.right() + 1, y, img, markerpixel)
            
    def point(self, x, y, img, pixel):
        if (x >= 0 and y >= 0):
            img[y][x] = pixel

    def path_to_processed_file(self, filename, classifier):
        return os.path.splitext(filename)[0] + '_' + classifier + os.path.splitext(filename)[1]

PORT = 8000

with socketserver.TCPServer(("", PORT), HTTPRequestHandler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()