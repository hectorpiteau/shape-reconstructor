import matplotlib.pyplot as plt
import cv2

class ImageEntity:
    image = None
    landmarks = []

    def __init__(self, image_path, landmarks_path) -> None:
        self.image = cv2.imread(image_path)
        self.landmarks = self.read_landmarks(landmarks_path)
        pass

    def read_landmarks(self, file_path):
        result = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split(' ')
                x = float(tokens[0])
                y = float(tokens[1])
                result.append({"x":x,"y":y})
        return result

    def show(self):
        for landmark in self.landmarks:
            # print("land: ",landmark)
            self.image = cv2.circle(self.image, (int(landmark["x"]), int(landmark["y"])), radius=2, color=(255,255,0), thickness=-1)
        plt.imshow(self.image)
        plt.show()
