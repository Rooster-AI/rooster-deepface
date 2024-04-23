
import gc
from pathlib import Path
import sys
import os
import cv2
from mtcnn import MTCNN
import pickle


def iterate_subdirectories(base_directory):
    for root, dirs, files in os.walk(base_directory):
        for directory in dirs:
            yield os.path.join(root, directory)

def find_images(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check the file extension and collect image files
            if os.path.splitext(file)[1].lower() in image_extensions:
                full_path = os.path.join(root, file)
                image_files.append(full_path)
                # print(f"Image found: {full_path}")
    return image_files

def create_labels(dir, teacher_model):

    skip_folders = {
        'train': [
            "0--Parade",
            "1--Handshaking",
            "2--Demonstration",
            "3--Riot",
            "4--Dancing",
            "5--Car_Accident",
            "6--Funeral",
            "7--Cheering",
            "8--Election_Campain",
            "9--Press_Conference",
            "10--People_Marching",
            "11--Meeting",
            "12--Group",
            "13--Interview",
            "14--Traffic",
            "15--Stock_Market",
            "16--Award_Ceremony",
            "17--Ceremony",
            "18--Concerts",
            "19--Couple",
            "20--Family_Group",
            # "21--Festival",
            # "22--Picnic",
            # "23--Shoppers",
            # "24--Soldier_Firing",
            # "25--Soldier_Patrol",
            # "26--Soldier_Drilling",
            # "27--Spa",
            # "28--Sports_Fan",
            # "29--Students_Schoolkids",
            # "30--Surgeons",
            # "31--Waiter_Waitress",
            # "32--Worker_Laborer",
            # "33--Running",
            # "34--Baseball",
            # "35--Basketball",
            # "36--Football",
            # "37--Soccer",
            # "38--Tennis",
            # "39--Ice_Skating",
            # "40--Gymnastics",
            # "41--Swimming",
            # "42--Car_Racing",
            # "43--Row_Boat",
            # "44--Aerobics",
            # "45--Balloonist",
            # "46--Jockey",
            # "47--Matador_Bullfighter",
            # "48--Parachutist_Paratrooper",
            # "49--Greeting",
            # "50--Celebration-Or-Party",
            # "51--Dresses",
            # "52--Photographers",
            # "53--Raid",
            # "54--Rescue",
            # "55--Sports_Coach-Trainer",
            # "56--Voter",
            # "57--Angler",
            # "58--Hockey",
            # "59--People--driving--car",
            # "61--Street_Battle",
        ],
        'test': [
            "0--Parade",
            "1--Handshaking",
            "2--Demonstration",
            "3--Riot",
            "4--Dancing",
            "5--Car_Accident",
            "6--Funeral",
            "7--Cheering",
            "8--Election_Campain",
            "9--Press_Conference",
            "10--People_Marching",
            "11--Meeting",
            "12--Group",
            "13--Interview",
            "14--Traffic",
            "15--Stock_Market",
            "16--Award_Ceremony",
            "17--Ceremony",
            "18--Concerts",
            "19--Couple",
            "20--Family_Group",
            "21--Festival",
            "22--Picnic",
            "23--Shoppers",
            "24--Soldier_Firing",
            "25--Soldier_Patrol",
            "26--Soldier_Drilling",
            "27--Spa",
            "28--Sports_Fan",
            "29--Students_Schoolkids",
            "30--Surgeons",
            "31--Waiter_Waitress",
            "32--Worker_Laborer",
            "33--Running",
            "34--Baseball",
            "35--Basketball",
            "36--Football",
            "37--Soccer",
            "38--Tennis",
            "39--Ice_Skating",
            "40--Gymnastics",
            "41--Swimming",
            "42--Car_Racing",
            "43--Row_Boat",
            # "44--Aerobics",
            # "45--Balloonist",
            # "46--Jockey",
            # "47--Matador_Bullfighter",
            # "48--Parachutist_Paratrooper",
            # "49--Greeting",
            # "50--Celebration-Or-Party",
            # "51--Dresses",
            # "52--Photographers",
            # "53--Raid",
            # "54--Rescue",
            # "55--Sports_Coach-Trainer",
            # "56--Voter",
            # "57--Angler",
            # "58--Hockey",
            # "59--People--driving--car",
            # "61--Street_Battle",
        ],
        'val': [
            "0--Parade",
            "1--Handshaking",
            "2--Demonstration",
            "3--Riot",
            "4--Dancing",
            # "5--Car_Accident",
            # "6--Funeral",
            # "7--Cheering",
            # "8--Election_Campain",
            # "9--Press_Conference",
            "10--People_Marching",
            "11--Meeting",
            "12--Group",
            "13--Interview",
            "14--Traffic",
            "15--Stock_Market",
            "16--Award_Ceremony",
            "17--Ceremony",
            "18--Concerts",
            "19--Couple",
            "20--Family_Group",
            "21--Festival",
            "22--Picnic",
            "23--Shoppers",
            "24--Soldier_Firing",
            "25--Soldier_Patrol",
            "26--Soldier_Drilling",
            "27--Spa",
            "28--Sports_Fan",
            "29--Students_Schoolkids",
            "30--Surgeons",
            "31--Waiter_Waitress",
            "32--Worker_Laborer",
            "33--Running",
            "34--Baseball",
            "35--Basketball",
            "36--Football",
            "37--Soccer",
            "38--Tennis",
            "39--Ice_Skating",
            "40--Gymnastics",
            "41--Swimming",
            "42--Car_Racing",
            "43--Row_Boat",
            "44--Aerobics",
            "45--Balloonist",
            "46--Jockey",
            "47--Matador_Bullfighter",
            "48--Parachutist_Paratrooper",
            "49--Greeting",
            "50--Celebration-Or-Party",
            "51--Dresses",
            "52--Photographers",
            "53--Raid",
            "54--Rescue",
            "55--Sports_Coach-Trainer",
            "56--Voter",
            "57--Angler",
            "58--Hockey",
            "59--People--driving--car",
            "61--Street_Battle",
        ]
    }

    base_directory = 'C:/Users/bridg/tensorflow_datasets/my_wider_face'

    base_directory = base_directory + '/' + dir

    label_dir = "/data/" + dir + '_labels/'

    print(f"Creating labels for {base_directory}")

    for directory in iterate_subdirectories(base_directory):
        print(f"Processing directory {directory}")
        folder = os.path.split(directory)[-1]
        print(f"Folder: {folder}")
        # print(f"Skip folders: {skip_folders[dir]}")
        if folder in skip_folders[dir]:
            print(f"Skipping {folder}")
            continue
        image_files = find_images(directory)
        for image_path in image_files:
            base_path = os.path.splitext(image_path)[0]
            img = cv2.imread(image_path)
            path_parts = base_path.split(os.sep)

            cwd = os.getcwd()

            parts = cwd + label_dir + path_parts[-1]

            pkl_path = parts + '.pkl'
            normalized_path = os.path.normpath(pkl_path)


            print(f"Processing {normalized_path}")

            # print(f"Processing {normalized_path}")
            detections = teacher_model.detect_faces(img)

            # # Append the file name 'pkl.pkl' to the current directory path
            with open(normalized_path, 'wb') as file:
                pickle.dump(detections, file)

            gc.collect()
            

if __name__ == '__main__':
    # get dir from args
    dir = sys.argv[1]
    teacher_model = MTCNN()
    create_labels(dir, teacher_model)