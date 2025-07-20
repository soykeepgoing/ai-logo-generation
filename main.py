import glob 
from tqdm import tqdm 
import cv2 
import pandas as pd 
from factor.shape import extract_symmetry, extract_roundness, extract_elaborate, extract_proportion, extract_repetition
from factor.color import extract_color_count, extract_main_color, extract_saturation_percent

def get_all_factors(image_files):
    data = {}
    
    for image_file in tqdm(image_files): 
        image = cv2.imread(image_file)
        image_idx = image_file.split('\\')[-1].replace('.jpg', '')

        # extract shape 
        sym = extract_symmetry(image)
        round = extract_roundness(image)
        elab = extract_elaborate(image)
        prop = extract_proportion(image)
        rep = extract_repetition(image)

        # extract color 
        color_cnt = extract_color_count(image)
        main_color = extract_main_color(image)
        saturation = extract_saturation_percent(image)

        data[image_idx] = [sym, round, elab, prop, rep, color_cnt, main_color, saturation]
    
    return data

if __name__ == "__main__":
    folder_name = "samples"
    image_files = sorted(glob.glob(f'{folder_name}\*.jpg')) 
    data = get_all_factors(image_files = image_files)
    df = pd.DataFrame(data).T
    df.columns = ['Symmetry', 'Roundness', 'Elaborate', 'Proportion', 'Repetition', 'ColorCnt', 'MainColor/Ratio', 'Saturation']

    df.to_csv('factors.csv')