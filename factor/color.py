import cv2 
import numpy as np 

color_dict_cnt = {
    'black': [[180, 255, 30], [0, 0, 0]],
    'white': [[180, 18, 255], [0, 0, 231]],
    'gray': [[180, 18, 230], [0, 0, 40]],
    'red': [[15, 255, 255], [0, 50, 70]],
    'red2': [[180, 255, 255], [165, 50, 70]],
    'yellow': [[45, 255, 255], [15, 50, 70]],
    'green': [[75, 255, 255], [45, 50, 70]],
    'cyan': [[105, 255, 255], [75, 50, 70]],
    'blue': [[135, 255, 255], [105, 50, 70]],
    'magenta': [[165, 255, 255], [135, 50, 70]],   
}

color_dict = {
    'black': [[180, 255, 50], [0, 0, 0]],
    'white': [[180, 50, 255], [0, 0, 200]],
    'gray': [[180, 50, 199], [0, 0, 51]],
    'red': [[30, 255, 255], [0, 50, 70]],
    'red2': [[180, 255, 255], [150, 50, 70]],
    'green': [[90, 255, 255], [30, 50, 70]],
    'blue': [[150, 255, 255], [90, 50, 70]],
    
}

def identify_color_from_HSV(hsv_value, color_dict):
    for color, (upper_bound, lower_bound) in color_dict.items():
        upper_bound_hue, upper_bound_saturation, upper_bound_value = upper_bound
        lower_bound_hue, lower_bound_saturation, lower_bound_value = lower_bound
        hue, saturation, value = hsv_value
        
        if (
            (lower_bound_hue <= hue < upper_bound_hue)
            and lower_bound_saturation <= saturation < upper_bound_saturation
            and lower_bound_value <= value < upper_bound_value
        ):
            return color
    
    return 'unknown'

def extract_color_count(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv_flatten = img_hsv.reshape(-1,3)
    img_hsv_flatten_uniq = np.unique(img_hsv_flatten,axis = 0)

    color = {}

    for hsv_value in img_hsv_flatten_uniq:
        color_detected = identify_color_from_HSV(hsv_value, color_dict_cnt)
        if color_detected != 'unknown':
            if color_detected == 'red2':
                color_detected = 'red'
            try: 
                color[color_detected] += 1
            except:
                color[color_detected] = 1
    return len(color)


def extract_main_color(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv_flatten = img_hsv.reshape(-1,3)
    img_hsv_flatten_uniq = np.unique(img_hsv_flatten,axis = 0)

    color = {}

    for hsv_value in img_hsv_flatten_uniq:
        color_detected = identify_color_from_HSV(hsv_value, color_dict)
        if color_detected != 'unknown':
            if color_detected == 'red2':
                color_detected = 'red'
            try: 
                color[color_detected] += 1
            except:
                color[color_detected] = 1

    max_key = max(color, key=color.get)
    total_counts = sum(color.values())
    max_ratio = round(color[max_key] / total_counts, 3)
    
    return (max_key, max_ratio)

def identify_saturation_from_HSV(hsv_value, color_dict):
    for color, (upper_bound, lower_bound) in color_dict.items():
        upper_bound_hue, upper_bound_saturation, upper_bound_value = upper_bound
        lower_bound_hue, lower_bound_saturation, lower_bound_value = lower_bound
        hue, saturation, value = hsv_value
        
        if (
            (lower_bound_hue <= hue < upper_bound_hue)
            and lower_bound_saturation <= saturation < upper_bound_saturation
            and lower_bound_value <= value < upper_bound_value
        ):
            
            if color not in ['black', 'white', 'gray']:
                return saturation
            
    return 'unknown'

def extract_saturation_percent(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv_flatten = img_hsv.reshape(-1,3)
    img_hsv_flatten_uniq = np.unique(img_hsv_flatten,axis = 0)

    saturations = []

    for hsv_value in img_hsv_flatten_uniq:
        saturation = identify_saturation_from_HSV(hsv_value, color_dict)
        if saturation != 'unknown':
            saturations.append(saturation)
            
    try: 
        saturation_percent = max(saturations) / 255
    except:
        saturation_percent = 0
        
    return round(saturation_percent,3)