import cv2
import numpy as np
import os
# from art import tprint
from random import randrange
from matplotlib import pyplot as plt


def search_new_file():
    path = '/Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Result/input'  # Путь к вашей папке

    # Получим список имен всего содержимого папки
    # и превратим их в абсолютные пути
    dir_list = [os.path.join(path, x) for x in os.listdir(path)]

    if dir_list:
        # Создадим список из путей к файлам и дат их создания.
        date_list = [[x, os.path.getctime(x)] for x in dir_list]

        # Отсортируем список по дате создания в обратном порядке
        sort_date_list = sorted(date_list, key=lambda x: x[1], reverse=True)

        # Выведем первый элемент списка. Он и будет самым последним по дате
        return (sort_date_list[0][0])


def apply_yolo_object_detection(image_to_process):
    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)  # горизонталь
                center_y = int(obj[1] * height)  # вертикаль
                obj_width = int(obj[2] * width)  # сжатие(горизонталь)
                obj_height = int(obj[3] * height)  # сжатие(вертикаль)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image_to_process = draw_object_bounding_box(image_to_process,
                                                        class_index, box)

    final_image = draw_object_count(image_to_process, objects_count)

    return final_image


def draw_object_bounding_box(image_to_process, index, box):
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (255, 0, 0)
    width = 2
    # print("image_to_process", image_to_process)
    print("start", start)
    print("end", end)
    # print("color", color)
    # print("width", width)
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    # x[0] = start[0]*1.25
    # start[0]=startCoordinatX
    # x[1] = start[1]*1.25
    # start[1] = startCoordinatY
    # y[1] = end[1]*1.10
    # end[1]=endCoordinatX
    # y[0] = end[0]*1.10
    # end[0]=endCoordinatY

    # print(startCoordinatX)
    # print(randrange(1000))
    randomnum = randrange(100000)
    print(randomnum)
    # print("Last added image: ", search_new_file())
#  /Users/georgijpustovalov/Downloads/webProjectCNN_jspCollab-master-2/src/main/webapp/downloadImages/loadFiles/download_images/bus1.png
    img = cv2.imread(search_new_file())

    # crop_img = img[start[0]-100:end[1]+150, start[1]+100:end[0]+100]
    crop_img = img[start[1]+100:end[1]+100, start[0]+100:end[0]+100]
    # crop_img = img[start[0]:end[0], start[0]:end[0]]


    # crop_img = img[startCoordinatX:endCoordinatX, startCoordinatY:endCoordinatY]
    stringpath = '/Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Result/output/' + str(
        randomnum) + '.jpg'
    stringpathcontur = '/Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Result/output/' + str(
        randomnum) + '.jpg'
    # cv2.imwrite(stringpath, crop_img)
    # cv2.imwrite(stringpathcontur, final_image)

    print("Successfuly added image")


    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font,
                              font_size, color, width, cv2.LINE_AA)



    return final_image


def draw_object_count(image_to_process, objects_count):
    start = (10, 150)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "[dtc0372_" + str(objects_count) + "_04]"



    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)



    return final_image


def start_image_object_detection(img_path):
    """
    Image analysis
    """
    print("img_path " + img_path)

    try:
        image = cv2.imread(img_path)
        image = apply_yolo_object_detection(image)
        cv2.imshow("Your image analitycs", image)
        # cv2.imwrite("/Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Result/output/" + image)

        if cv2.waitKey(0):
            cv2.destroyAllWindows()
            # cv2.imwrite("/Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Result/output/"+ image)
            # isWritten = cv2.imwrite("/Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Result/output/" + image)

    except KeyboardInterrupt:
        pass





def color_detected(img_path):
    img = cv2.imread(img_path)
    img_red = cv2.imread(img_path)
    img_yellow = cv2.imread(img_path)
    # image = cv2.resize(img, (700, 600))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)
    hsv_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2HSV)

    # /Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Result/input/rgb_test.png

    # traffic light

    # зеленый цвет
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([100, 255, 255])


    # красный цвет
    lower_red = np.array([10, 100, 100])
    upper_red = np.array([10, 255, 255])

    # желтый цвет
    lower_yellow = np.array([29, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask_red = cv2.inRange(hsv_red, lower_red, upper_red)

    mask_yellow = cv2.inRange(hsv_yellow, lower_yellow, upper_yellow)


    kernel_green = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel_green)

    kernel_yellow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_yellow)

    kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_red)




    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour_green in contours_green:
        x, y, w, h = cv2.boundingRect(contour_green)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, 'Green light', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


    for contour_yellow in contours_yellow:
      x, y, w, h = cv2.boundingRect(contour_yellow)
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
      cv2.putText(img, 'Yellow light', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)



    for contour_red in contours_red:
        x, y, w, h = cv2.boundingRect(contour_red)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, 'Red light', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()






    # plt.imshow(cv2.cvtColor(img_yellow, cv2.COLOR_BGR2RGB))
    # plt.show()
    #
    #
    # kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_red)
    #
    #
    #
    # plt.imshow(cv2.cvtColor(img_red, cv2.COLOR_BGR2RGB))
    # plt.show()

















    # mask = cv2.bitwise_or(mask_green, mask_red)
    # result = cv2.bitwise_and(image, image, mask=mask)
    #
    # cv2.imshow('Result', result)
    #

    # GREEN_light = cv2.boundingRect(mask_green)
    # # traffic light
    # if GREEN_light is not None:
    #     print("Green light detected")
    #     x, y, w, h = GREEN_light
    #     # x, y, w, h = cv2.boundingRect(GREEN_light)
    #     image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     # cv2.putText(image, 'Green light', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    #     # cv2.putText("Green light")
    # else:
    #     print("Green light not detected")
    # #
    # cv2.imshow('image', image)
    # / Users / georgijpustovalov / Downloads / yolodetected_generatescript - main / Result / input / trafic_light_test.png


    # cv2.imshow('image', mask_green)
    # cv2.imshow('image', mask_red)
    # cv2.imshow('image', mask_yellow)

    # RED_light = cv2.boundingRect(mask_red)
    # # traffic light
    # if RED_light is not None:
    #     print("Red light detected")
    #     x1, y1, w1, h1 = RED_light
    #     # x, y, w, h = cv2.boundingRect(GREEN_light)
    #     image = cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    #     cv2.putText(image, 'Red light', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    #     # cv2.putText("Green light")
    #     # cv2.imshow('image', img1)
    # else:
    #     print("Red light not detected")



    # YELLOW_light = cv2.boundingRect(mask_yellow)
    # # traffic light
    # if YELLOW_light is not None:
    #     print("Yellow light detected")
    #     x2, y2, w2, h2 = YELLOW_light
    #     # x, y, w, h = cv2.boundingRect(GREEN_light)
    #     image = cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
    #     cv2.putText(image, 'Yellow light', (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    #     # cv2.putText("Green light")
    #     # cv2.imshow('image', img2)
    # else:
    #     print("Yellow light not detected")


    cv2.waitKey(0)










if __name__ == '__main__':

    net = cv2.dnn.readNetFromDarknet("/Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Resources/yolov4-tiny.cfg",
                                     "/Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    with open("/Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    image = input("Enter path to image location: ")

    # image = search_new_file()

    look_for = input("Enter what object detected: ").split(',')
    # look_for = 'traffic light'
    # /Users/georgijpustovalov/Downloads/yolodetected_generatescript-main/Result/input/trafic_light_test.png


    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for

    start_image_object_detection(image);

    color_detected(image);








