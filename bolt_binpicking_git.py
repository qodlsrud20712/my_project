import os
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
from multiprocessing import freeze_support
from detectron2.utils.visualizer import ColorMode
import cv2

# labelme2coco 모듈로 labelme로 라벨링한 데이터들을 coco형식의 데이터셋 json파일로 만들어줘야된다.
# import labelme2coco
#
# input = r"C:\Users\a\PycharmProjects\rect_test"
# out = "./rect.json"
#
# labelme2coco.convert(input, out)

# freeze_support()를 써주는 이유 : 윈도우 환경에서 발생하는 동시성? 문제인거같다.
# if __name__ == "__main__": 요거 밑에 같이 써주어야 된다.
# 다른 해결책들은 https://aigong.tistory.com/136 이곳을 참고하면될듯.
freeze_support()
#밑에 세줄 코드를 작성하여 데이터셋을 해당 데이터셋을 만들어주어야한다.
# test2 = register_coco_instances("small_bolt", {}, "./test.json", r"C:\Users\a\Downloads\lr_1\base")
# dataset_dicts_2 = DatasetCatalog.get("small_bolt")
# my_dataset_train_metadata_2 = MetadataCatalog.get("_")

# cfg 파일을 만들어야한다. yaml 파일은 이미 작성되어있는 거대한 데이터 파일이다.
# 이것을 이용해서 내 모델을 훈련시켜야 한다.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("small_bolt2",)
cfg.DATASETS.TRAIN = ("small_bolt",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = r"C:\Users\a\PycharmProjects\detectorn2\output\model_final.pth"
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# 요건 트레이닝할 때 쓰이는 코드 , 저번 훈련 때 부터 다시할려면 resume를 true로 weights는 쓰지않는다. max_iter는 저번보다 높게 지정해야된다.
#     # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#     # trainer = DefaultTrainer(cfg)
#     # trainer.resume_or_load(resume=True)
#     # trainer.checkpointer.load(r"C:\Users\pjs\PycharmProjects\detectron2\output\model_final.pth")
#     # trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                 "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

# 실제 사용할 때 경로(지금 계획은 사진한장을 찍어 저장시킨 후 detection할 예정. 어떻게 선명하게 저장할지
# 선명하게 저장하기 위해선 프레임이 몇번 돌아야 된다. -> 해결 쓰레드로 사진 선명하게 저장 완료
# path_dir = r"C:\Users\a\PycharmProjects\detectorn2\data\object_detect.png"
# file_list = os.listdir(path_dir)
# 테스트용 경로
# path_dir = r"C:\Users\a\PycharmProjects\lr_1\base\many_1.jpg"
path_dir = r"C:\Users\a\PycharmProjects\detectorn2\rgb.jpg"

im = cv2.imread(path_dir)
outputs = predictor(
    im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
v = Visualizer(im[:, :, ::-1],
               metadata=None,
               scale=0.7,
               instance_mode=ColorMode.IMAGE_BW
               # remove the colors of unsegmented pixels. This option is only available for segmentation models
               )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
box_data = outputs["instances"].to("cpu")

# 이 코드로 box data를 넘겨받아서 array로 넘겨줌.
tmp = box_data.pred_boxes.tensor.numpy()
# numpy array를 list로
tmp = tmp.tolist()
# 5개의 object seg 좌표가 들어있는 리스트(visualizer.py, draw_polygon에서 나옴)
img_p_list = v.segment