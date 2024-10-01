#!/usr/bin/python3

import gi
import torch
import numpy as np
import time

from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.net_utils import load_network

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Lanedet config
cfg = Config.fromfile("configs/laneatt/resnet18_culane.py")
cfg.load_from = "checkpoints/laneatt_r18_culane.pth"
cfg.ori_img_w = 1280
cfg.ori_img_h = 510

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
            self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, ori_img):
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data


detect = Detect(cfg)


def filter_noise_from_line(line, threshold=10):
    x = line[:, 0]
    y = line[:, 1]

    # Fit a line (1st degree polynomial) to the data
    slope, intercept = np.polyfit(x, y, 1)

    # Calculate the distance of each point to the fitted line
    # Distance formula for a point (x0, y0) to a line ax + by + c = 0 is |ax0 + by0 + c| / sqrt(a^2 + b^2)
    distances = np.abs(slope * x - y + intercept) / np.sqrt(slope ** 2 + 1)

    # Filter points based on the threshold
    filtered_indices = distances < threshold
    filtered_line = line[filtered_indices]

    return filtered_line

def calculate_rotation_angle(lines, image_width, threshold=10):
    # Filter noise for each line
    filtered_lines = [filter_noise_from_line(line, threshold) for line in lines]

    # Calculate the distance of each line's average x position to the vertical center (image_width / 2)
    center_x = image_width / 2
    distances_to_center = [np.abs(np.mean(line[:, 0]) - center_x) for line in filtered_lines]

    # Find indices of the two lines closest to the center
    closest_indices = np.argsort(distances_to_center)[:2]

    # Extract the two closest lines
    line1, line2 = filtered_lines[closest_indices[0]], filtered_lines[closest_indices[1]]

    # Calculate the middle line between the two center lines
    # Ensure that both lines have the same number of points
    if len(line1) != len(line2):
        # Interpolate to match the number of points
        max_len = max(len(line1), len(line2))
        line1_interp = np.column_stack((
            np.interp(np.linspace(0, len(line1) - 1, max_len), np.arange(len(line1)), line1[:, 0]),
            np.interp(np.linspace(0, len(line1) - 1, max_len), np.arange(len(line1)), line1[:, 1])
        ))
        line2_interp = np.column_stack((
            np.interp(np.linspace(0, len(line2) - 1, max_len), np.arange(len(line2)), line2[:, 0]),
            np.interp(np.linspace(0, len(line2) - 1, max_len), np.arange(len(line2)), line2[:, 1])
        ))
    else:
        line1_interp = line1
        line2_interp = line2

    # Calculate the middle line by averaging corresponding points
    middle_line = (line1_interp + line2_interp) / 2

    def calculate_angle(line):
        # Perform linear regression (1st order polynomial fitting)
        x = line[:, 0]
        y = line[:, 1]

        # Fit a first-degree polynomial (line) to the data
        slope, intercept = np.polyfit(x, y, 1)

        # Angle in radians (from vertical axis, hence arctan(1/m))
        angle_rad = np.arctan(1 / slope)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    angle = calculate_angle(middle_line)

    return angle, [line1, line2, middle_line]

def process_frame(sample):
    # Convert GStreamer buffer to numpy array
    buffer = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value('width')
    height = structure.get_value('height')
    array = np.ndarray(
        (height, width, 3),
        buffer=buffer.extract_dup(0, buffer.get_size()),
        dtype=np.uint8
    )

    data = detect.preprocess(array)
    start = time.time()
    data['lanes'] = detect.inference(data)[0]
    end = time.time()
    lanes = [lane.to_array(cfg) for lane in data['lanes']]
    try:
        angle, lanes = calculate_rotation_angle(lanes, cfg.ori_img_w)
        print(f"angle: {angle}, time: {end - start}")
    except:
        print("proc err")

def on_frame(appsink):
    sample = appsink.emit('pull-sample')
    process_frame(sample)
    return Gst.FlowReturn.OK

# Create the GStreamer pipeline
pipeline = Gst.parse_launch(
    # 'filesrc location=/home/filipp/Downloads/center2.mkv ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink'
    'v4l2src device=/dev/video0 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink'
)

# Get the appsink element
appsink = pipeline.get_by_name('sink')
appsink.set_property('emit-signals', True)
appsink.connect('new-sample', on_frame)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Run the main loop
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pass

# Clean up
pipeline.set_state(Gst.State.NULL)
