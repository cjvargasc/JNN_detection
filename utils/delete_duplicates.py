"""

Quick fix to remove duplicated lines saved into mAP files

"""
import os

if __name__ == "__main__":
    paths = ["/home/mmv/Documents/2.projects/Object-Detection-Metrics-master/detections/",
             "/home/mmv/Documents/2.projects/Object-Detection-Metrics-master/groundtruths/"]

    for path in paths:
        for file in os.listdir(path):
            if file.endswith(".txt"):
                lines_seen = set()
                f = open(path + file, "r")
                lines = f.readlines()
                for each_line in lines:
                    if each_line not in lines_seen:  # check if line is not duplicate
                        lines_seen.add(each_line)
                f.close()

                f = open(path + file, "w")
                for each_line in lines_seen:
                    f.write(each_line)
                f.close()






