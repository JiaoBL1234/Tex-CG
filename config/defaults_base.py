from yacs.config import CfgNode as CN



_C = CN()

_C.MODEL = CN()

_C.MODEL.DEVICE = "cuda"

_C.MODEL.DEVICE_ID = '0'

_C.MODEL.NAME = 'resnet50'

_C.MODEL.LAST_STRIDE = 1

_C.MODEL.PRETRAIN_PATH = ''

_C.MODEL.PRETRAIN_CHOICE = 'imagenet'


_C.MODEL.NECK = 'bnneck'
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS = True
_C.MODEL.I2T_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_CLOTH_LOSS = False
_C.MODEL.I2T_CLOTH_LOSS_WEIGHT = 1.0
_C.MODEL.STAGE2_I2T_LOSS = False
_C.MODEL.STAGE2_I2T_CLOTH_LOSS = False
_C.MODEL.STAGE2_I2T_LOSS_WEIGHT = 1.0
_C.MODEL.STAGE2_I2T_CLOTH_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'

_C.MODEL.DIST_TRAIN = False

_C.MODEL.NO_MARGIN = False

_C.MODEL.IF_LABELSMOOTH = 'on'

_C.MODEL.COS_LAYER = False


_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]


_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

_C.MODEL.CLOTH_PROMPT = False
_C.MODEL.ID_PROMPT = False

_C.INPUT = CN()

_C.INPUT.SIZE_TRAIN = [384, 128]

_C.INPUT.SIZE_TEST = [384, 128]

_C.INPUT.PROB = 0.5

_C.INPUT.RE_PROB = 0.5

_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]

_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

_C.INPUT.PADDING = 10


_C.DATASETS = CN()

_C.DATASETS.NAMES = ('market1501')

_C.DATASETS.ROOT_DIR = ('../data')



_C.DATALOADER = CN()

_C.DATALOADER.NUM_WORKERS = 8

_C.DATALOADER.SAMPLER = 'softmax'

_C.DATALOADER.NUM_INSTANCE = 16


_C.SOLVER = CN()

_C.SOLVER.OPTIMIZER_NAME = "Adam"

_C.SOLVER.MAX_EPOCHS = 100

_C.SOLVER.BASE_LR = 3e-4

_C.SOLVER.LARGE_FC_LR = False

_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.SEED = 1234

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MARGIN = 0.3

_C.SOLVER.CENTER_LR = 0.5

_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005


_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005


_C.SOLVER.GAMMA = 0.1

_C.SOLVER.STEPS = (40, 70)

_C.SOLVER.WARMUP_FACTOR = 0.01

_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_LR_INIT = 0.01
_C.SOLVER.LR_MIN = 0.000016


_C.SOLVER.WARMUP_ITERS = 500

_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30


_C.SOLVER.CHECKPOINT_PERIOD = 10

_C.SOLVER.LOG_PERIOD = 100

_C.SOLVER.EVAL_PERIOD = 10

_C.SOLVER.IMS_PER_BATCH = 64



_C.TEST = CN()

_C.TEST.IMS_PER_BATCH = 128

_C.TEST.RE_RANKING = False

_C.TEST.WEIGHT = ""

_C.TEST.NECK_FEAT = 'after'

_C.TEST.FEAT_NORM = 'yes'


_C.TEST.DIST_MAT = "dist_mat.npy"

_C.TEST.EVAL = False

_C.OUTPUT_DIR = ""
