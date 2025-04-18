
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import Levenshtein
import tensorflow as tf  

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image augmentation and batch data loading

from tensorflow.keras.models import Sequential  # For stacking layers linearly to build a simple CNN
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPool2D, 
                                     BatchNormalization, Flatten, GlobalAveragePooling2D, Input)

from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam  # Adaptive optimizer for faster convergence

from tensorflow.keras.callbacks import (
    EarlyStopping,         # Stops training when validation loss stops improving
    ModelCheckpoint,       # Saves the best-performing model during training
    ReduceLROnPlateau,     # Reduces learning rate when a metric stops improving
    LearningRateScheduler  # Custom learning rate schedule per epoch
)

from tensorflow.keras.models import load_model


def directory_to_df(path : str):


    df = []

    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"    # to include lowercase letters only

    #chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:,.#%'-()"    # to include lowercase letters only
    for cls in os.listdir(path):
        #print(cls)
        cls_path = os.path.join(path,cls)
        #print(cls_path)
        if "sym" in cls_path:
            cls_name = cls.split('_')[1]
        else:
            cls_name = cls.split('_')[0]
        #print(cls_name)
        if not cls_name in chars:
            #print("oh no")
            continue
        for img_path in os.listdir(cls_path):
            direct = os.path.join(cls_path,img_path)
            df.append([direct,cls_name])
    
    df = pd.DataFrame(df, columns=['image','label'])
    #print("The number of samples found:",len(df))
    return df.copy()

def read_image(path):

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_image(img, label=None) -> None:

    plt.imshow(img, cmap='gray')
    plt.axis(False)
    plt.title(label)
    plt.show()
    


IMG_SHAPE = (32,32)
IMG_SIZE = (32,32,3)
BATCH_SIZE = 32
opt = Adam(learning_rate=0.00001, epsilon=1e-6)
loss = 'categorical_crossentropy'

# Reading the dataset in dataframe 
main_path = 'datasetold'
df = directory_to_df(main_path)  

# Splitting for training & testing (70,30 respectively)
X, y = df['image'], df['label']
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.30, random_state=41)
training_df = pd.concat((X_train,y_train), axis=1)
testing_df = pd.concat((X_test,y_test), axis=1)

X, y = training_df['image'], training_df['label']
X_train, X_valid, y_train, y_valid = train_test_split(X,y , test_size=0.25, random_state=41)
training_df = pd.concat((X_train,y_train), axis=1)
validation_df = pd.concat((X_valid,y_valid), axis=1)


# Creating generators
gen = ImageDataGenerator(dtype=np.int32, brightness_range=[0.0,1.0], fill_mode='nearest')
gen2 = ImageDataGenerator(dtype=np.int32, fill_mode='nearest')
train_gen = gen.flow_from_dataframe(training_df, x_col='image',y_col='label', batch_size=BATCH_SIZE, 
                                   target_size=IMG_SHAPE)
valid_gen = gen2.flow_from_dataframe(validation_df, x_col='image', y_col='label', batch_size=BATCH_SIZE, 
                                        target_size=IMG_SHAPE, shuffle=False)
test_gen = gen2.flow_from_dataframe(testing_df, x_col='image', y_col='label', batch_size=BATCH_SIZE, 
                                       target_size=IMG_SHAPE, shuffle=False)

mapping = train_gen.class_indices
mapping_inverse = dict(map(lambda x: tuple(reversed(x)), mapping.items()))



# Computer Vision - Low level techniques
def load_models():
    model_path = 'custom_cnn_model_all.keras'
    model = load_model(model_path)
    return model

def convert_2_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def binarization(image):
    img, thresh = cv2.threshold(image, 0,255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    return img, thresh

def dilate(image, words= False):
    img = image.copy()
    m = 3
    n = m - 2                   # n less than m for Vertical structuring element to dilate chars
    itrs = 4
    if words:
        m = 6
        n = m
        itrs = 3
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))
    dilation = cv2.dilate(img, rect_kernel, iterations = itrs)
    return dilation

def find_rect(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = []
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)  # Extract the bounding rectangle coordinates of each countour
        rects.append([x,y,w,h])
        
    sorted_rects = list(sorted(rects, key=lambda x: x[0])) # Sorting the rects from Left-to-Right
    return sorted_rects


def extract(image):
    model = load_models()
    chars = []              # a list to store recognized characters
    
    image_cpy = image.copy()
    _, bin_img = binarization(convert_2_gray(image_cpy))
    full_dil_img = dilate(bin_img,words=True)
    words = find_rect(full_dil_img)                       # Recognized words within the image 
    del _, bin_img, full_dil_img                          # for better memory usage
    
    for word in words:
        x,y,w,h = word                                    # coordinates of the word
        img = image_cpy[y:y+h, x:x+w]
        
        _, bin_img = binarization(convert_2_gray(img))
        dil_img = dilate(bin_img)
        char_parts = find_rect(dil_img)                     # Recognized chars withtin the word
        cv2.rectangle(image, (x,y),(x+w,y+h), (0,255,0), 3) # draw a green rectangle around the word
        
        
        for char in char_parts:    
            x,y,w,h = char
            ch = img[y:y+h, x:x+w]
            
            empty_img = np.full((32,32,1),255, dtype=np.uint8) # a white image used for resize with filling
            x,y = 3,3                                          # starting indecies
            resized = cv2.resize(ch, (16,22), interpolation=cv2.INTER_CUBIC)
            gray = convert_2_gray(resized)
            empty_img[y:y+22, x:x+16,0] = gray.copy()          # integrate the recognized char into the white image
            gray = cv2.cvtColor(empty_img, cv2.COLOR_GRAY2RGB)
            gray = gray.astype(np.int32)
            
            predicted = mapping_inverse[np.argmax(model.predict(np.array([gray]), verbose=-1))]
            chars.append(predicted)                            # append the character into the list
            
        chars.append(' ')  # at the end of each iteration (end of word) append a space
        
    show_image(image)
    return ''.join(chars[:-1])

if len(sys.argv) != 2:
    print("Usage: python script.py <image_filename>")
    sys.exit(1)

image_file = sys.argv[1]
img = read_image(image_file)
text = extract(img)
text = text.lower()
#print('OCR Output -->', text)

ocr_words = text.split(" ")
print("OCR Extracted Raw Words --> ",ocr_words)

df_food = pd.read_csv('FoodData.csv')
df_food['Food'] = df_food['Food'].str.lower()
df_food_unique = df_food.drop_duplicates(subset=['Food'], keep='first')
food_dict = df_food_unique.set_index('Food').to_dict('index')

df_complex_names = pd.read_csv('ComplexIngredients.csv')
df_complex_names['Complex_Ingredient'] = df_complex_names['Complex_Ingredient'].str.lower()
complex_names = df_complex_names.set_index('Complex_Ingredient').to_dict('index')

#Complex_Ingredient,Simple_Synonym

def trim_candidates(word):
    """
    Generate a set of candidate words by removing one or two characters
    from the beginning or the end of the given word.
    """
    candidates = [word]  # include the original word
    if len(word) > 1:
        candidates.append(word[:-1])            # remove last char
        candidates.append(word[1:])             # remove first char
    if len(word) > 2:
        candidates.append(word[:-2])            # remove last two chars
        candidates.append(word[2:])             # remove first two chars
        candidates.append(word[1:-1])           # remove first and last char
    return candidates

def generate_ambiguous_variants(word, ambiguous_mapping=None):
    """
    Generate all variants of a word by replacing ambiguous characters.
    
    The ambiguous_mapping argument should be a dictionary where each key 
    maps to a list of alternative characters.
    """
    if ambiguous_mapping is None:
        ambiguous_mapping = {
            'i': ['l', 'j'],
            'l': ['i'],
            'r': ['f', 't'],
            'f': ['r'],
            'j': ['i'],
            'q': ['a'],
            't': ['r']
        }
    
    results = set()
    
    def helper(idx, current):
        if idx == len(word):
            results.add(current)
        else:
            char = word[idx]
            # Always include the original character.
            helper(idx + 1, current + char)
            # If there are ambiguous replacements for this char, include them.
            if char in ambiguous_mapping:
                for alt in ambiguous_mapping[char]:
                    helper(idx + 1, current + alt)
    
    helper(0, "")
    return results



def generate_candidates(word):
    cand_set = trim_candidates(word)
    final_candidates = set()
    for candidate in cand_set:
        print(candidate)
        print()
        final_candidates.update(generate_ambiguous_variants(candidate))
    
    # Define a custom sorting key:
    # 1. Prioritize candidates equal to the original word.
    # 2. Then, sort by the absolute difference in length from the original word.
    # 3. Finally, lexicographical order.
    def sort_key(candidate):
        return (candidate != word, abs(len(candidate) - len(word)), candidate)
    
    return sorted(final_candidates.union(cand_set), key=sort_key)


def get_best_match(word, dictionary, max_dist=1):
    """
    Find and return the key in the dictionary that best matches the given word,
    according to the Levenshtein distance. If the best distance is within the 
    max_dist threshold, return the matching word; otherwise return None.
    """
    best_match = None
    best_distance = float('inf')
    for dict_word in dictionary.keys():
        distance = Levenshtein.distance(word, dict_word)
        if distance < best_distance:
            best_distance = distance
            best_match = dict_word
    if best_distance <= max_dist:
        return best_match
    else:
        return None


results = []

for ocr_word in ocr_words:
    ocr_word = ocr_word.lower()  # standardize to lower case
    candidates = generate_candidates(ocr_word)
    match_found = None
    # For each candidate, try to find a good match in the dictionary.
    for candidate in candidates:
        candidate = candidate.lower()  # ensure candidate is lowercase
        match = get_best_match(candidate, food_dict, max_dist=1)
        if match:
            match_found = match
            break  # Use the first candidate that meets our condition.
    
    if match_found:
        # Fetch additional information from the dictionary.
        if match_found[0] != ocr_word[0] and len(match_found) < len(ocr_word):
            continue
        if len(ocr_word) - len(match_found) >=2:
            continue
        info = food_dict[match_found]
        results.append({
            'OCR_word': ocr_word,
            'Matched_Food': match_found,
            'Class': info.get('Class'),
            'Type': info.get('Type'),
            'Group': info.get('Group'),
            'Allergy': info.get('Allergy')
        })
    else:
        continue


results2 = []

for ocr_word in ocr_words:
    ocr_word = ocr_word.lower()  # standardize to lower case
    candidates = generate_candidates(ocr_word)
    match_found = None
    # For each candidate, try to find a good match in the dictionary.
    for candidate in candidates:
        if("mai" in candidate or "mal" in candidate):
            print(candidate)
        candidate = candidate.lower()  # ensure candidate is lowercase
        match = get_best_match(candidate, complex_names, max_dist=1)
        if match:
            match_found = match
            break  # Use the first candidate that meets our condition.
    
    if match_found:
        # Fetch additional information from the dictionary.
        if match_found[0] != ocr_word[0] and len(match_found) < len(ocr_word):
            continue
        if len(ocr_word) - len(match_found) >=2:
            continue

        info = complex_names[match_found]
        results2.append({
            'OCR_word': ocr_word,
            'Complex Ingredient': match_found,
            'Simpler Synonym': info.get('Simple_Synonym'),
        })
    else:
        continue

# -----------------------------------------------------
# Step 4. Display the results.
# -----------------------------------------------------

if results:
    print("\n\nAllergens Warning:")

for result in results:
    if result['Matched_Food'] is not None:
        print(f"OCR word '{result['OCR_word']}' matched with FoodData CSV entry '{result['Matched_Food']}'.")
        print(f"  Class : {result['Class']}")
        print(f"  Type  : {result['Type']}")
        print(f"  Group: {result['Group']}")
        print(f"  Allergy: {result['Allergy']}\n")

if results2:
    print("\n\nMapping Complex Ingredients to simpler synonyms:")

for result in results2:
    if result['Complex Ingredient'] is not None:
        print(f"OCR word '{result['OCR_word']}' matched with Complex Ingredient entry '{result['Complex Ingredient']}'.")
        print(f"  Complex Ingredient : {result['Complex Ingredient']}")
        print(f"  Simpler Synonym  : {result['Simpler Synonym']}\n")

