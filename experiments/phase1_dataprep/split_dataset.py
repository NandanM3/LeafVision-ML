import os           #work with files and directories
import shutil       #copy and move files
import random       #generate random numbers to reorder files

SOURCE_DIR = 'ML\Tomato_Leaf_Data'               #original dataset
OUTPUT_DIR = 'ML\Tomato_Leaf_Data_Split'         #split dataset


TRAIN_RATIO = 0.7  
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  #set seed for reproducibility

def create_folder(path):    #create a folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

def split_class(class_name):    
    class_path = os.path.join(SOURCE_DIR, class_name)     
    images = os.listdir(class_path)

    random.shuffle(images)  #shuffles the images to ensure randomness in the split

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split_name, split_images in zip(
        ["train", "val", "test"],
        [train_images, val_images, test_images]
    ):
        split_class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
        create_folder(split_class_dir)

        for image in split_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(split_class_dir, image)
            shutil.copyfile(src, dst)

    print(class_name, "done!")
    print(f"Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")
    print("----------")


def main():
    create_folder(OUTPUT_DIR)

    classes = os.listdir(SOURCE_DIR)

    for class_name in classes:
        split_class(class_name)


if __name__ == "__main__":
    main()



