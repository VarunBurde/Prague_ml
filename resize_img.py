import cv2


img = cv2.imread("/Users/artemmoroz/CIIRC_work/Prague_ml/visulization_data/tang_dist_2.png")
img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_AREA)
cv2.imwrite("/Users/artemmoroz/CIIRC_work/Prague_ml/visulization_data/tang_dist_2.png", img)

