""" 
This program identifies the lane on a road in real time. It is based on the GeeksforGeeks version of the code linked below.

GeeksforGeeks Code: https://www.geeksforgeeks.org/opencv-real-time-road-lane-detection/

By: Nividh Singh
Date: 9/25/2023


"""

# Libraries for working with image processing
import numpy as np
import pandas as pd
import cv2

# Libraries needed to edit/save/watch video clips
from moviepy import editor
import moviepy


# This function processes the video. It analyzes each frame and outputs it to the output file
def process_video(test_video, output_video):
    input_video = editor.VideoFileClip(test_video, audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(output_video, audio=False)
    
# This function analyzes a sepcific image (which is a frame from the video). 
def frame_processor(image):
    
    # Converts to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blurs the image
    kernal_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernal_size, kernal_size), 0)
    
    # Identifies the edges
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    
    #Because there are a lot of edges in the image, this finds the region where the road is
    region = region_selection(edges)
    
    # This finds the two lines that are on the edges of the lane
    hough = hough_transform(region)
    
    # Adds the line on top of the original frame and returns result
    result = draw_lane_lines(image, lane_lines(image, hough))
    return result

# Selects region where the lane is
def region_selection(image):
    
    
    mask = np.zeros_like(image)
    
    # Passes image with more than one channel (grayscale is one channel image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        
    else:
        
        ignore_mask_color = 255
        
    # Finds the polygon for the road (based on where the camera was plaecd)
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
    # Fills polygon with white to generate final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # Finds edges of the road using bitwise and
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

# Does a hough tansformation using cv2's function. The hough transformation is used to find basic geometric figures like lines and circles
def hough_transform(image):
    
    rho = 1
    theta = np.pi/180
    threshold = 20
    minLineLength = 20
    maxLineGap = 500
    
    return cv2.HoughLinesP(image, rho = rho, theta=theta, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

# Finds average slope of left and right line
def average_slope_intercept(lines):
    
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    
    # Goes through each line
    for line in lines:
        
        # Goes through each x and y
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            
            # If there are two different points, we find the slope and intercept
            slope = (y2 - y1) / (x2 - x1)
            
            intercept = y1 - (slope * x1)
            
            length = np.sqrt(((y2-y1) ** 2) + ((x2 - x1) ** 2))
            
            # If it goes inward, add to left line lists
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
                
            # If it goes inward on the right side, add to right line lists
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
                
    # Create lanes based on the weights for each line
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

# Converts slope-intercept  of each line into a pixel point
def pixel_points(y1, y2, line):
    
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

# Create full length lines from pixel points
def lane_lines(image, lines):
    
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

# Draws line on top of image
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
    
 


# calling driver function
process_video('test2.mp4','output.mp4')